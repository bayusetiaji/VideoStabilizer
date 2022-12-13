// Video Stabilizer Implementation
// ref: https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv/

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct TransformParam
{
	//TransformParam() {}

	TransformParam(double _dx, double _dy, double _da)
	{
		dx = _dx;
		dy = _dy;
		da = _da;
	}

	void getTransform(Mat& T)
	{
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);

		T.at<double>(0, 2) = dx;
		T.at<double>(1, 2) = dy;
	}

	double dx;
	double dy;
	double da;
};

struct Trajectory
{
	Trajectory(double _x, double _y, double _a)
	{
		x = _x;
		y = _y;
		a = _a;
	}

	double x;
	double y;
	double a;
};

vector<Trajectory> cumsum(vector<TransformParam>& transforms)
{
	vector<Trajectory> trajectory;

	// accumulated frame to frame transform
	double a = 0;
	double x = 0;
	double y = 0;

	for (size_t i = 0; i < transforms.size(); ++i)
	{
		x += transforms[i].dx;
		y += transforms[i].dy;
		a += transforms[i].da;

		trajectory.push_back(Trajectory(x, y, a));
	}

	return trajectory;
}

vector<Trajectory> smooth(vector<Trajectory>& trajectory, int radius)
{
	vector<Trajectory> smoothed_trajectory;
	for (size_t i = 0; i < trajectory.size(); ++i)
	{
		double sum_x = 0;
		double sum_y = 0;
		double sum_a = 0;
		int count = 0;

		for (int j = -radius; j <= radius; ++j)
		{
			if (i + j >= 0 && i + j < trajectory.size())
			{
				sum_x += trajectory[i + j].x;
				sum_y += trajectory[i + j].y;
				sum_a += trajectory[i + j].a;

				count++;
			}
		}

		double avg_x = sum_x / count;
		double avg_y = sum_y / count;
		double avg_a = sum_a / count;

		smoothed_trajectory.push_back(Trajectory(avg_x, avg_y, avg_a));
	}

	return smoothed_trajectory;
}

void fixBorder(Mat& frame_stabilized)
{
	Mat T = getRotationMatrix2D(Point(frame_stabilized.cols / 2, frame_stabilized.rows / 2), 0, 1.04);
	warpAffine(frame_stabilized, frame_stabilized, T, frame_stabilized.size());
}

int main()
{
	const int SMOOTHING_RADIUS = 1;
	
	// 1. read and write video
	VideoCapture vid("D:/vids/sketches/sample.mp4");
	
	int frameCount = (int)vid.get(CAP_PROP_FRAME_COUNT);
	int frameW = (int)vid.get(CAP_PROP_FRAME_WIDTH);
	int frameH = (int)vid.get(CAP_PROP_FRAME_HEIGHT);
	double fps = vid.get(CAP_PROP_FPS);
	
	VideoWriter vout("D:/vids/sketches/vout.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(2 * frameW, frameH));

	// 2. read the first frame and convert to grayscale
	Mat curr, curr_gray;
	Mat prev, prev_gray;

	vid >> prev;

	cvtColor(prev, prev_gray, COLOR_BGR2GRAY);

	// 3. find motion between frames
	vector<TransformParam> transforms;

	Mat last_T;
	for (int i = 1; i < frameCount; ++i)
	{
		vector<Point2f> prev_pts, curr_pts;

		// detect feature in previous frame
		goodFeaturesToTrack(prev_gray, prev_pts, 200, 0.01, 30);

		// read next frame
		bool succes = vid.read(curr);
		if (!succes)
			break;

		// conver to grayscale
		cvtColor(curr, curr_gray, COLOR_BGR2GRAY);

		// calculate optical flow (i.e. track feature points)
		vector<uchar> status;
		vector<float> err;
		calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, curr_pts, status, err);

		// filter only valid points
		auto prev_it = prev_pts.begin();
		auto curr_it = curr_pts.begin();
		for (size_t k = 0; k < status.size(); ++k)
		{
			if (status[k])
			{
				prev_it++;
				curr_it++;
			}
			else
			{
				prev_it = prev_pts.erase(prev_it);
				curr_it = curr_pts.erase(curr_it);
			}
		}

		// find transformation matrix
		// Mat T = estimateRigidTransform(prev_pts, curr_pts, false);
		Mat T = estimateAffinePartial2D(prev_pts, curr_pts);

		// in rare cases no transform is found
		// we'll just use the last known good transform
		if (T.data == NULL)
			last_T.copyTo(T);
		T.copyTo(last_T);

		// extract translation
		double dx = T.at<double>(0, 2);
		double dy = T.at<double>(1, 2);

		// extract rotation angle
		double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));

		// store transformation
		transforms.push_back(TransformParam(dx, dy, da));

		// move to next frame
		curr_gray.copyTo(prev_gray);

		cout << "Frame: " << i << "/" << frameCount << " - tracked points: " << prev_pts.size() << endl;
	}

	// 4. calculate smooth motion between frames
	auto trajectory = cumsum(transforms);
	vector<Trajectory> smoothed_trajectory = smooth(trajectory, SMOOTHING_RADIUS);

	vector<TransformParam> transforms_smooth;
	for (size_t i = 0; i < transforms.size(); ++i)
	{
		// calculate difference in smoothed_trajectory and trajectory
		double diff_x = smoothed_trajectory[i].x - trajectory[i].x;
		double diff_y = smoothed_trajectory[i].y - trajectory[i].y;
		double diff_a = smoothed_trajectory[i].a - trajectory[i].a;

		// calculate newer transformation array
		double dx = transforms[i].dx + diff_x;
		double dy = transforms[i].dy + diff_y;
		double da = transforms[i].da + diff_a;

		transforms_smooth.push_back(TransformParam(dx, dy, da));
	}

	// 5. apply smoothed camera motion to frames
	vid.set(CAP_PROP_POS_FRAMES, 1);
	Mat T(2, 3, CV_64F);
	Mat frame, frame_stabilized, frame_out;

	for (int i = 0; i < frameCount - 1; ++i)
	{
		bool success = vid.read(frame);
		if (!success)
			break;

		// extract transform from transation and rotation angle
		transforms_smooth[i].getTransform(T);

		// apply affine wrapping to the given frame
		warpAffine(frame, frame_stabilized, T, frame.size());

		// scale image to remove black border artifact
		fixBorder(frame_stabilized);

		// draw the original and stabilized side by side for coolness
		hconcat(frame, frame_stabilized, frame_out);

		// if the image is too big, resize it
		if (frame_out.cols > 1920)
			resize(frame_out, frame_out, Size(frame_out.cols / 2, frame_out.rows / 2));

		imshow("Before and After", frame_out);
		vout.write(frame_out);
		waitKey(10);
	}

	vid.release();
	destroyAllWindows();
	
	return 0;
}
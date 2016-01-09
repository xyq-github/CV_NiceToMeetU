#include <iostream>
#include <sstream>
#include <ctime>
#include <cstdio>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class Settings
{
public:
    Settings() : goodInput(false) {}
    enum Pattern {NOT_EXISTING, CHESSBOARD};	// modified
    enum InputType { INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST };

    void write(FileStorage &fs) const
    {
        fs << "{"
                << "BoardSize_Width" << boardSize.width
                << "BoardSize_Height" << boardSize.height
                << "Square_Size" << squareSize
                << "Calibrate_Pattern" << patternToUse
                << "Calibrate_NrOfFrameToUse" << nrFrames
                << "Calibrate_FixAspectRatio" << aspectRatio
                << "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
                << "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint
                
                << "Write_DetectedFeaturePoints" << writePoints
                << "Write_extrinsicParameters" << writeExtrinsics
                << "Write_outputFileName" << outputFileName

                << "Show_UndistortedImage" << showUndistorted

                // << "Input_FilpAroundHorizontalAxis" << flipVertical
                << "Input_Delay" << delay
                << "Input" << input
            << "}";
    }

    void read(const FileNode& node) 
    {
        node["BoardSize_Width"] >> boardSize.width;
        node["BoardSize_Height"] >> boardSize.height;
        node["Calibrate_Pattern"] >> patternToUse;
        node["Square_Size"] >> squareSize;
        node["Calibrate_NrOfFrameToUse"] >> nrFrames;
        node["Calibrate_FixAspectRatio"] >> aspectRatio;
        node["Write_DetectedFeaturePoints"] >> writePoints;
        node["Write_extrinsicParameters"] >> writeExtrinsics;
        node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
        node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
        // node["Input_FileAroundHorizontalAxis"] >> flipVertical;
        node["Show_UndistortedImage"] >> showUndistorted;
        node["Input"] >> input;
        node["Input_Delay"] >> delay;
        validate();
    }
    
    void validate()
    {
        goodInput = true;
        if (boardSize.width <= 0 || boardSize.height <= 0)
        {
            goodInput = false;
            cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
        }
        if (squareSize <= 10e-6)
        {
            cerr << "Invalid square size: " << squareSize << endl;
            goodInput = false;
        }
        if (nrFrames <= 0)
        {
            cerr << "Invalid number of frames: " << nrFrames << endl;
            goodInput = false;
        }
        if (input.empty())
            inputType = INVALID;
        else
        {
            if (input[0] >= '0' && input[0] <= '9')
            {
                stringstream ss(input);
                ss >> cameraID;
                inputType = CAMERA;
            }
            else
            {
                if (readStringList(input, imageList))
                {
                    inputType = IMAGE_LIST;
                    nrFrames = min(nrFrames, (int)imageList.size());
                }
                else
                    inputType = VIDEO_FILE;
            }
            if (inputType == CAMERA)
                inputCapture.open(cameraID);
            if (inputType == VIDEO_FILE)
                inputCapture.open(input);
            if (inputType != IMAGE_LIST && !inputCapture.isOpened())
                inputType = INVALID;
        }
        if (inputType == INVALID)
        {
            cerr << "Input does not exist: " << input;
            goodInput = false;
        }

        // flag = CALIB_FIX_K4 | CALIB_FIX_K5;
        if (calibFixPrincipalPoint) flag |= CALIB_FIX_PRINCIPAL_POINT;
        if (calibZeroTangentDist)   flag |= CALIB_ZERO_TANGENT_DIST;
        if (aspectRatio)			flag |= CALIB_FIX_ASPECT_RATIO;

        calibrationPattern = NOT_EXISTING;
        if (!patternToUse.compare("CHESSBOARD"))
            calibrationPattern = CHESSBOARD;
        if (calibrationPattern == NOT_EXISTING)
        {
            cerr << "Camera calibration mode does not exist: " << patternToUse << endl;
            goodInput = false;
        }
        atImageList = 0;
    }

    Mat nextImage()
    {
        Mat result;
        if (inputCapture.isOpened())
            inputCapture >> result;
        else if (atImageList < imageList.size())
            result = imread(imageList[atImageList++], IMREAD_COLOR);

        return result;
    }

    static bool readStringList(const string& filename, vector<string>& l)
    {
        l.clear();
        FileStorage fs(filename, FileStorage::READ);
        if (!fs.isOpened())
            return false;
        FileNode n = fs.getFirstTopLevelNode();
        if (n.type() != FileNode::SEQ)
            return false;
        FileNodeIterator it = n.begin(), it_end = n.end();
        for (; it != it_end; ++it)
            l.push_back((string)*it);
        return true;
    }
public:
    Size boardSize;
    Pattern calibrationPattern;
    float squareSize;
    int nrFrames;
    float aspectRatio;
    int delay;
    bool writePoints;
    bool writeExtrinsics;
    bool calibFixPrincipalPoint;
    bool calibZeroTangentDist;
    // bool flipVertical;
    string outputFileName;
    bool showUndistorted;
    string input;
    
    int cameraID;
    vector<string> imageList;
    size_t atImageList;
    VideoCapture inputCapture;
    InputType inputType;
    bool goodInput;
    int flag;

private:
    string patternToUse;
};

static inline void read(const FileNode& node, Settings& x, const Settings& default_value=Settings())
{
	if (node.empty())
		 x = default_value;
	else
		x.read(node);
}

static inline void write(FileStorage& fs, const string&, const Settings& s)
{
	s.write(fs);
}

enum { DETECTION = 0, CAPTURING, CALIBRATED };

bool runCalibrationAndSave( Settings& s, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
							vector<vector<Point2f> > imagePoints );

int main(int argc, char **argv)
{
	Settings s;
	const string inputSettingsFile = argc > 1 ? argv[1] : "default.xml";
	FileStorage fs(inputSettingsFile, FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file: " << inputSettingsFile << endl;
		return -1;
	}
	fs["Settings"] >> s;
	fs.release();

	FileStorage fout("settings.yml", FileStorage::WRITE);
	fout << "Settings" << s;

	if (!s.goodInput)
	{
		cout << "Invalid input detected. Application stopping." << endl;
		return -1;
	}

	vector< vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	Size imageSize;
	int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
	clock_t prevTimestamp = 0;
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0);
	const char ESC_KEY = 27;

	for (;;)
	{
		Mat view;
		// bool blinkOutput = false;

		view = s.nextImage();

        // got enough picutes OR reached an empty picutre, then calibrate
		if (mode == CAPTURING && imagePoints.size() >= (size_t)s.nrFrames)
		{
			if ( runCalibrationAndSave(s, imageSize, cameraMatrix, distCoeffs, imagePoints))
				mode = CALIBRATED;
			else
				mode = DETECTION;
		}
		if (view.empty())
		{
			if (mode != CALIBRATED &&!imagePoints.empty())
				runCalibrationAndSave(s, imageSize, cameraMatrix, distCoeffs, imagePoints);
			break;
		}

		imageSize = view.size();
		// if (s.flipVertical) flip(view, view, 0);

		vector<Point2f> pointBuf;

		bool found;

        // flags description on opencv docs
        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

		switch (s.calibrationPattern)
		{
		case Settings::CHESSBOARD:
			found = findChessboardCorners( view, s.boardSize, pointBuf, chessBoardFlags);
			break;
		default :
			found = false;
			break;
		}

		if (found)
		{
			if (s.calibrationPattern == Settings::CHESSBOARD)
			{
				Mat viewGray;
				cvtColor(view, viewGray, COLOR_BGR2GRAY);
				cornerSubPix(viewGray, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 30, 0.1));
			}

			if (mode == CAPTURING &&
				(!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC) )
			{
				imagePoints.push_back(pointBuf);
				prevTimestamp = clock();
				// blinkOutput = s.inputCapture.isOpened();
			}

			drawChessboardCorners( view, s.boardSize, Mat(pointBuf), found);
		}

		string msg = (mode == CAPTURING) ? "100/100" : 
					mode == CALIBRATED ? "Calibrated" : "Press g to start";
		int baseLine = 0;
		Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
		Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

		if (mode == CAPTURING)
		{
			if (s.showUndistorted)
				msg = format( "%d/%d Undist", (int)imagePoints.size(), s.nrFrames);
			else
				msg = format( "%d/%d", (int)imagePoints.size(), s.nrFrames);
		}

		putText(view, msg, textOrigin, 1, 1, mode == CALIBRATED ? GREEN : RED);

		// if (blinkOutput)
		// 		bitwise(view, view);

		if (mode == CALIBRATED && s.showUndistorted)
		{
			Mat temp = view.clone();
			undistort(temp, view, cameraMatrix, distCoeffs);
		}

		imshow("Image View", view);
		char key = (char)waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

		if (key == ESC_KEY)
			break;

		if (key == 'u' && mode == CALIBRATED)
			s.showUndistorted = !s.showUndistorted;

		if (s.inputCapture.isOpened() && key == 'g')
		{
			mode = CAPTURING;
			imagePoints.clear();
		}
	} // for;

	if (s.inputType == Settings::IMAGE_LIST && s.showUndistorted)
	{
		Mat view, rview;

		for (size_t i = 0; i < s.imageList.size(); ++i)
		{
			view = imread(s.imageList[i], 1);
			if (view.empty())
				continue;
			undistort(view, rview, cameraMatrix, distCoeffs);
			imshow("Image View", rview);
			char c = (char)waitKey(100);
			if (c == ESC_KEY || c == 'q' || c == 'Q')
				break;
		}
	}

	return 0;
}

static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
										const vector<vector<Point2f> >& imagePoints,
										const vector<Mat>& rvecs,
										const vector<Mat>& tvecs,
										const Mat& cameraMatrix,
										const Mat& distCoeffs,
										vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	size_t totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (size_t i = 0; i < objectPoints.size(); ++i)
	{
		projectPoints(objectPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
		err = norm(imagePoints[i], imagePoints2, NORM_L2);

		size_t n = objectPoints[i].size();
		perViewErrors[i] = (float) sqrt(err*err/n);
		totalErr += err*err;
		totalPoints += n;
	}

	return sqrt(totalErr/totalPoints);
}

static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners, Settings::Pattern patternType = Settings::CHESSBOARD)
{
	corners.clear();
	switch (patternType)
	{
		case Settings::CHESSBOARD:
			for (int i = 0; i < boardSize.height; ++i)
				for (int j = 0; j < boardSize.width; ++j)
					corners.push_back(Point3f(j*squareSize, i*squareSize, 0));
			break;
		default:
			break;
	}
}

static bool runCalibration(Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs, vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs, vector<float>& reprojErrs, double& totalAvgErr)
{
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (s.flag & CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = s.aspectRatio;
	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);
	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	double rms;
	rms = calibrateCamera(objectPoints, imagePoints,
                          imageSize, cameraMatrix,
                          distCoeffs, rvecs, tvecs, s.flag);

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;
	
	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
                                            rvecs, tvecs,
                                            cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

static void saveCameraParams(Settings& s, Size& imageSize,
                            Mat& cameraMatrix, Mat& distCoeffs,
                            const vector<Mat>& rvecs, const vector<Mat>& tvecs,
							const vector<float>& reprojErrs,
							const vector<vector<Point2f> >& imagePoints,
							double totalAvgErr)
{
	FileStorage fs(s.outputFileName, FileStorage::WRITE);

	time_t tm;
	time(&tm);
	struct tm *t2 = localtime(&tm);
	char buf[1024];
	strftime(buf, sizeof(buf), "%c", t2);

	fs << "calibration_time" << buf;

	if (!rvecs.empty() || !reprojErrs.empty())
		fs << "nr_of_frames" << (int)max(rvecs.size(), reprojErrs.size());
	fs << "image_width" << imageSize.width;
	fs << "image_height" << imageSize.height;
	fs << "board_width" << s.boardSize.width;
	fs << "board_height" << s.boardSize.height;
	fs << "square_size" << s.squareSize;

	if (s.flag & CALIB_FIX_ASPECT_RATIO)
		fs << "fix_aspect_ratio" << s.aspectRatio;

	if (s.flag)
	{
		sprintf(buf, "flags:%s%s%s%s",
				s.flag & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
				s.flag & CALIB_FIX_ASPECT_RATIO ? "+fix_aspect_ratio" : "",
				s.flag & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
				s.flag & CALIB_ZERO_TANGENT_DIST ?  " +zero_rangent_dist" : "");
		cvWriteComment(*fs, buf, 0);
	}

	fs << "flags" << s.flag;

	fs << "camera_matrix" << cameraMatrix;
	fs << "distortion_coefficients" << distCoeffs;
	
	fs << "avg_reprojction_error" << totalAvgErr;
	if (s.writeExtrinsics && !reprojErrs.empty())
		fs << "per_view_reprojection_errors" << Mat(reprojErrs);

	if (s.writeExtrinsics && !rvecs.empty() && !tvecs.empty())
	{
		CV_Assert(rvecs[0].type() == tvecs[0].type());
		Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
		for (size_t i = 0; i < rvecs.size(); ++i)
		{
			Mat r = bigmat(Range(int(i), int(i+1)), Range(0, 3));
			Mat t = bigmat(Range(int(i), int(i+1)), Range(3, 6));

			CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
			CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);

			r = rvecs[i].t();
			t = tvecs[i].t();
		}
        fs << "extrinsic_parameters" << bigmat;
	}

	if (s.writePoints && !imagePoints.empty())
	{
		Mat imagePtMat((int)imagePoints.size(), (int)imagePoints[0].size(), CV_32FC2);
		for (size_t i = 0; i < imagePoints.size(); ++i)
		{
			Mat r = imagePtMat.row(int(i)).reshape(2, imagePtMat.cols);
			Mat imgpti(imagePoints[i]);
			imgpti.copyTo(r);
		}
		fs << "image_points" << imagePtMat;
	}
}

bool runCalibrationAndSave(Settings& s, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs, vector<vector<Point2f> > imagePoints)
{
	vector<Mat> rvecs, tvecs;
	vector<float> reprojErrs;
	double totalAvgErr = 0;

	bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs, reprojErrs, totalAvgErr);
	cout << (ok ? "Calibration succeeded" : "Calibration failed")
		<< ". avg reprojection error = " << totalAvgErr << endl;

	if (ok)
		saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, reprojErrs, imagePoints, totalAvgErr);

	return ok;
}


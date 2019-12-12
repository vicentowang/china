#define _CRT_SECURE_NO_WARNINGS
//stdout
#include <iostream>

//stl
#include <vector>
#include <map>

//c++11 thread
#include <thread>
#include <mutex>

//opencv
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\tracking.hpp>
#include <opencv2\xfeatures2d.hpp>

//camera driver
#include <io.h>
#include <windows.h>
#include <direct.h>
#include <conio.h>
#include <MvCameraControl.h>


using namespace std;
using namespace cv;
using namespace xfeatures2d;
mutex mts_master;
mutex diff_mxt;
mutex mark_mxt;

#define CHECK_EXP_RETURN(x)  if(x) return
#define CHECK_EXP_RETURN(x,y)  if(x) return y

void img_pre_processing(Mat & src)
{
	if (src.empty()) return;
	if (src.type() != CV_8UC1) src.convertTo(src, CV_8UC1);
	threshold(src, src, 0, 255, THRESH_OTSU);
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(src, src, MORPH_OPEN, element, Point(-1, -1), 1);
	morphologyEx(src, src, MORPH_CLOSE, element, Point(-1, -1), 1);
}

Mat imrotate(Mat img, int degree)
{
	degree = -degree;

	double angle = degree * CV_PI / 180.; // 弧度  

	double a = sin(angle), b = cos(angle);

	int width = img.cols;

	int height = img.rows;

	int width_rotate = int(height * fabs(a) + width * fabs(b));

	int height_rotate = int(width * fabs(a) + height * fabs(b));

	//旋转数组map

	// [ m0  m1  m2 ] ===>  [ A11  A12   b1 ]

	// [ m3  m4  m5 ] ===>  [ A21  A22   b2 ]

	float map[6];

	Mat map_matrix = Mat(2, 3, CV_32F, map);

	// 旋转中心

	CvPoint2D32f center = cvPoint2D32f(width / 2, height / 2);

	CvMat map_matrix2 = map_matrix;

	cv2DRotationMatrix(center, degree, 1.0, &map_matrix2);

	map[2] += (width_rotate - width) / 2;

	map[5] += (height_rotate - height) / 2;

	Mat img_rotate;

	//对图像做仿射变换

	//CV_WARP_FILL_OUTLIERS - 填充所有输出图像的象素。

	//如果部分象素落在输入图像的边界外，那么它们的值设定为 fillval.

	//CV_WARP_INVERSE_MAP - 指定 map_matrix 是输出图像到输入图像的反变换，

	warpAffine(img, img_rotate, map_matrix, Size(width_rotate, height_rotate), 1, 0, 0);

	return img_rotate;

}

vector<Point2f> re_mapping(const vector<Point2f> & X, const Mat & A)
{
	vector<Point2f> Y;
	for (auto x : X)
	{
		Mat X = (Mat_<double>(3, 1) << x.x, x.y, 1);
		Mat X_Ow = (A * X);
		Y.push_back(Point2f(X_Ow.at<double>(Point(0, 0)), X_Ow.at<double>(Point(0, 1))));
	}
	return Y;
}

Mat get_normed_affine_transform(const vector<Point2f> & src_tri ,bool obj_to_oc = true)
{
	Mat_<double> A;
	double scale_x = norm(src_tri[0] - src_tri[1]);
	double scale_y = norm(src_tri[0] - src_tri[2]);
	vector<Point2f> dst_trip;
	double edge = 0;
	dst_trip.push_back(Point2f(edge, edge));
	dst_trip.push_back(Point2f(edge + scale_x, edge));
	dst_trip.push_back(Point2f(edge, edge + scale_y));
	if (obj_to_oc)
	{
		A = getAffineTransform(dst_trip , src_tri);
	}
	else
	{
		A = getAffineTransform(src_tri , dst_trip);

		//Mat A_inv = getAffineTransform(src_tri, dst_trip);
		//vector<Point2f> dst = re_mapping(src_tri ,A_inv);
	}
	
	return A;
}

double angle(const Point2f & pt0 , const Point2f & pt1, const Point2f & pt2 )
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	double angle_line = (dx1*dx2 + dy1 * dy2) / sqrt((dx1*dx1 + dy1 * dy1)*(dx2*dx2 + dy2 * dy2) + 1e-10);
	return acos(angle_line) * 180 / 3.141592653;
}

double right_hand_rule(const Vec3f & v1 , const Vec3f & v2)
{
	return v1.cross(v2)[2];
}

void right_hand_based_tripple_sort(vector<Point2f> & src_trip)
{
	map<double,Point2f> p_map;
	size_t s = src_trip.size();
	const int trip_t = 3;
	for (int i = 0 ; i < src_trip.size() ; ++i)
	{
		p_map[angle(src_trip[(i) % s], src_trip[(i + 1) % s], src_trip[(i + 2) % s])] = src_trip[i];
	}
	src_trip.clear();
	CHECK_EXP_RETURN(p_map.size() != trip_t);

	auto it1 = p_map.begin();
	auto it2 = ++p_map.begin();
	auto it3 = --p_map.end();

#ifdef _DEBUG
	//cout << "tripple_angle : " << it3->first << " , " << it2->first << " , " << it1->first << endl;
#endif
	Point2f vertex = it3->second;
	Point2f mid = Point2f( (it1->second.x + it2->second.x)/2.0 , (it1->second.y + it2->second.y) / 2.0);
	double z_dir = right_hand_rule(Vec3f(mid.x - vertex.x, mid.y - vertex.y,0), Vec3f(it1->second.x - vertex.x, it1->second.y - vertex.y,0));
	CHECK_EXP_RETURN(z_dir == 0);

	src_trip.push_back(vertex);
	if ( z_dir < 0 )
	{
		src_trip.push_back( it1->second );
		src_trip.push_back( it2->second );
	}
	else if( z_dir > 0 )
	{
		src_trip.push_back(it2->second);
		src_trip.push_back(it1->second);
	}
	return;
}

void set_lable(Mat & im , const string label , const Point2f & p )
{
	int fontface = FONT_HERSHEY_SIMPLEX;
	double scale = 0.4;
	int thickness = 1;
	int baseline = 5;
	int bias = 10;
	Size text = getTextSize(label, fontface, scale, thickness, &baseline);
	//rectangle(im, p +Point2f(0, baseline), p + Point2f(text.width, -text.height), CV_RGB(0, 0, 255));
	putText(im, label, Point(p.x,p.y+bias), fontface, scale, CV_RGB(255, 255, 255), thickness, FILLED);
}

void draw_circle(Mat & img,const Mat & _set)
{
	auto dim = _set.channels();
	if (dim == 3)
	{
		for (auto it = _set.begin<Vec3f>(); it != _set.end<Vec3f>(); ++it)
		{
			Point s = Point2f((*it)[0], (*it)[1]);
			circle(img, s, 1, Scalar(200, 0, 0), 2, FILLED);
		}
	}
	else if (dim == 2)
	{
		for (auto it = _set.begin<Vec2f>(); it != _set.end<Vec2f>(); ++it)
		{
			Point s = Point2f((*it)[0], (*it)[1]);
			circle(img, s, 3, Scalar(200, 0, 0),2, LINE_8);
		}
	}
}

void draw_circle(Mat & img, vector<Point3f> & p_set)
{
	for (auto p : p_set)
	{
		Point s = Point( p.x , p.y);
		circle(img, s, 4, Scalar(128, 0, 0));
	}
}

void draw_circle(Mat & img, vector<Point2f> & p_set ,Scalar sl = Scalar(188, 0, 0))
{
	for (auto p : p_set)
	{
		Point s = Point2f(p);
		circle(img, s, 1, sl,2 , FILLED);
	}
}

vector<Point2f> get_centre2( const Mat & img_, const vector<Rect2f> & roi_trip)
{
	Mat img = img_;// .clone();
	if(img.channels() == 3) cvtColor(img, img, CV_BGR2GRAY);
	GaussianBlur(img, img, Size(3, 3), 0);
	threshold(img, img, 0, 255, CV_THRESH_OTSU);
	Mat _img(img.size(), CV_8UC1, Scalar(0));
	
	for (auto & roi : roi_trip)
	{
		_img(roi) = 255;
	}
	img.copyTo(_img, _img);

	Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
	morphologyEx(_img, _img, MORPH_OPEN, element, Point(-1, -1), 5);
	morphologyEx(_img, _img, MORPH_CLOSE, element, Point(-1, -1),5);
	_img.convertTo(_img, CV_8UC1);

	vector<vector<Point>> contours;
	cv::findContours(_img, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
#ifdef _DEBUG
    drawContours(_img, contours , -1 , Scalar(180), FILLED);
#endif
	vector<Moments> Mpq(contours.size());
	for (size_t i = 0; i < contours.size(); i++)
	{
		Mpq[i] = moments(contours[i], true);
	}
	vector<Point2f> trip_centre(Mpq.size());
	for (size_t i = 0; i < Mpq.size(); i++)
	{
		trip_centre[i] = Point2f(static_cast<float>(Mpq[i].m10 / Mpq[i].m00), static_cast<float>(Mpq[i].m01 / Mpq[i].m00));
	}
	return trip_centre;
}

Point2d get_centre(const Mat & img ,const Point2f & obj_tl)
{
	vector<vector<Point>> contours;
	cv::findContours(img, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	double area_total = 0;
	const double area_th = 0.0005;
	for (auto& c : contours)
	{
		area_total += moments(c, true).m00;
	}

	Point2f master_vec = Point2f(0, 0);
	vector<Point2f> avg_cen;
	for (auto & c : contours)
	{
		Moments Mpq = moments(c, true);
		double area_rate = Mpq.m00 / area_total;
		if (area_rate < area_th)
		{
			img(boundingRect(c)) = 0;
			continue;
		}
		Point2f centre = Point2f(static_cast<float>(Mpq.m10 / Mpq.m00), static_cast<float>(Mpq.m01 / Mpq.m00));
		avg_cen.push_back(centre * area_rate);
	}
	Point2f cen = Point2f(0, 0);
	for (size_t i = 0; i < avg_cen.size(); ++i)
	{
		cen += avg_cen[i];
	}
	return cen + obj_tl;
}

vector<Point2f> matching_template_accelerated_knn(const Mat & _img, const Mat & _mark, double scaler_f = 0.3, double similarity = 0.95, int k = 3, double roi_scaler = 0.2)
{
	Mat img, mark;
	cv::resize(_img, img, Size(), scaler_f, scaler_f, INTER_CUBIC);
	cv::resize(_mark, mark, Size(), scaler_f, scaler_f, INTER_CUBIC);

	Mat fus_map;
	vector<Point2f> src_trip;
	if (mark.rows > img.rows || mark.cols > img.cols)return src_trip;
	matchTemplate(img, mark, fus_map, TM_CCORR_NORMED);
	threshold(fus_map, fus_map, similarity, 255, THRESH_TOZERO);

	vector<Rect2f> roi_mask_trip, roi_trip;
	for (int i = 0; i < k; ++i)
	{
		double maxVal; Point maxLoc;
		minMaxLoc(fus_map, NULL, &maxVal, NULL, &maxLoc);
		if (maxVal <= 0) break;
		Point2f matchLoc = maxLoc;
		Rect2d roi = Rect2f(matchLoc, Size(mark.cols, mark.rows));

		Point2d scale_shift = (Point2d(roi.tl() - roi.br()) * roi_scaler);
		fus_map(Rect2d(roi.tl() - scale_shift, roi.br() + scale_shift) & Rect2d(0, 0, fus_map.cols, fus_map.rows)) = 0;

		roi = Rect2d(roi.tl() / scaler_f, roi.br() / scaler_f) & Rect2d(0, 0, _img.cols, _img.rows);
		roi_trip.push_back(roi);
	}
	Mat img_s = _img.clone();
	//src_trip = get_centre(img_s, roi_trip);
	right_hand_based_tripple_sort(src_trip);

	if (src_trip.size() == 3)
	{
		Point2f v1 = src_trip[1] - src_trip[0];
		Point2f v2 = src_trip[2] - src_trip[0];
		double  ang = acos(v1.dot(v2) / (norm(v1) * norm(v2))) * 180 / 3.1415926;
		if (fabs(ang - 90) > 1)
		{
			cout << "error ,tripple mark is not orthogonal ,rectify it !" << endl;
			Mat X = (Mat_<double>(3, 1) << src_trip[1].x, src_trip[1].y, 1); //防止3个mark 不正交
			X = (getRotationMatrix2D(src_trip[0], -90, 1) * X);
			src_trip[2] = Point2f(X.at<double>(Point(0, 0)), X.at<double>(Point(0, 1)));
		}
	}
	else
	{
		cout << "matching tripple mark failed !" << endl;
	}
	return src_trip;
}

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	//s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}

void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const vector<Point> &, Mat&);
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
	double angle = atan2((double)p.y - q.y, (double)p.x - q.x); // angle in radians
	double hypotenuse = sqrt((double)(p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
	// Here we lengthen the arrow by a factor of scale
	q.x = (int)(p.x - scale * hypotenuse * cos(angle));
	q.y = (int)(p.y - scale * hypotenuse * sin(angle));
	line(img, p, q, colour, 1, LINE_AA);
	// create the arrow hooks
	p.x = (int)(q.x + 9 * cos(angle + CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle + CV_PI / 4));
	line(img, p, q, colour, 1, LINE_AA);
	p.x = (int)(q.x + 9 * cos(angle - CV_PI / 4));
	p.y = (int)(q.y + 9 * sin(angle - CV_PI / 4));
	line(img, p, q, colour, 1, LINE_AA);
}

int getOrientation(const vector<Point2f> &pts, Mat &img)
{
	//Construct a buffer used by the pca analysis
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
	for (int i = 0; i < data_pts.rows; i++)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	//Perform PCA analysis
	PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
	//Store the center of the object
	Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
	//Store the eigenvalues and eigenvectors
	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}
	// Draw the principal components
	circle(img, cntr, 3, Scalar(255, 0, 255), 2);
	Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
	Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
	drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
	drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);
	double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x) * 180 / 3.1415926; // orientation in radians
	return angle;
}


Point2f getOrientation(const vector<Point2f> & pts, Mat &img, Point2f & cntr)
{
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
	for (int i = 0; i < data_pts.rows; i++)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW, 2);

	//cntr = Point2f(static_cast<float>(pca_analysis.mean.at<double>(0, 0)),
	//	static_cast<float>(pca_analysis.mean.at<double>(0, 1)));

	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}
	//double len_vec = norm(eigen_vecs[0]);
	// Draw the principal components
	circle(img, cntr, 2, Scalar(122, 122, 122), 1);
	Point2f p1 = Point2f(static_cast<float>(eigen_vecs[0].x * eigen_val[0]), static_cast<float>(eigen_vecs[0].y * eigen_val[0]));
	//len_vec = norm(p1);
	//Point2f p11 = p1 / len_vec;
	Point2f p2 = Point2f(static_cast<float>(eigen_vecs[1].x * eigen_val[1]), static_cast<float>(eigen_vecs[1].y * eigen_val[1]));
	Point2f p3 = (p1)+(p2);//Point2f(p1.x + p2.x, p1.y + p2.y);
	line(img, cntr, cntr + p1, Scalar(99, 99, 0), 1);
	line(img, cntr, cntr + p2, Scalar(99, 99, 0), 1);
	line(img, cntr, cntr + p3, Scalar(33, 33, 0), 1);
	double angle1 = atan2(p1.y, p1.x) * 180 / 3.1415926; // orientation in radians
	double angle2 = atan2(p2.y, p2.x) * 180 / 3.1415926; // orientation in radians
	double angle3 = atan2(p3.y, p3.x) * 180 / 3.1415926; // orientation in radians

	return p3;
}

Point2f getOrientation(const vector<Point> & pts, Mat &img, Point2f & cntr)
{
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
	for (int i = 0; i < data_pts.rows; i++)
	{
		data_pts.at<double>(i, 0) = pts[i].x;// -cntr.x;
		data_pts.at<double>(i, 1) = pts[i].y;// -cntr.y;
	}
	PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW, 2);

	cntr = Point2f(static_cast<float>(pca_analysis.mean.at<double>(0, 0)),
		static_cast<float>(pca_analysis.mean.at<double>(0, 1)));

	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}
	//double len_vec = norm(eigen_vecs[0]);
	// Draw the principal components
	//circle(img, cntr, 2, Scalar(122, 122, 122), 1);
	Point2f p1 = Point2f(static_cast<float>(eigen_vecs[0].x * eigen_val[0]), static_cast<float>(eigen_vecs[0].y * eigen_val[0]));
	//len_vec = norm(p1);
	//Point2f p11 = p1 / len_vec;
	//Point2f p2 = Point2f(static_cast<float>(eigen_vecs[1].x * eigen_val[1]), static_cast<float>(eigen_vecs[1].y * eigen_val[1]));
	Point2f p3 = (p1);//+(p2);//Point2f(p1.x + p2.x, p1.y + p2.y);
	//line(img, cntr, cntr + p1, Scalar(99, 99, 0), 1);
	//line(img, cntr, cntr + p2, Scalar(99, 99, 0), 1);
	//line(img, cntr, cntr + p3, Scalar(33, 33, 0), 1);
	//double angle1 = atan2(p1.y, p1.x) * 180 / 3.1415926; // orientation in radians
	//double angle2 = atan2(p2.y, p2.x) * 180 / 3.1415926; // orientation in radians
	//double angle3 = atan2(p3.y, p3.x) * 180 / 3.1415926; // orientation in radians

	return p3;
}

Point2f getOrientation2(const vector<Point> & pts, Mat &img, Point2f & cntr)
{
	int sz = static_cast<int>(pts.size());
	Mat data_pts = Mat(sz, 2, CV_64F);
	for (int i = 0; i < data_pts.rows; i++)
	{
		data_pts.at<double>(i, 0) = pts[i].x;
		data_pts.at<double>(i, 1) = pts[i].y;
	}
	PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW, 2);

	//cntr = Point2f(static_cast<float>(pca_analysis.mean.at<double>(0, 0)),
	//	static_cast<float>(pca_analysis.mean.at<double>(0, 1)));

	vector<Point2d> eigen_vecs(2);
	vector<double> eigen_val(2);
	for (int i = 0; i < 2; i++)
	{
		eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
			pca_analysis.eigenvectors.at<double>(i, 1));
		eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
	}

	circle(img, cntr, 2, Scalar(122, 122, 122), 1);
	Point2f p1 = Point2f(static_cast<float>(eigen_vecs[0].x * eigen_val[0]), static_cast<float>(eigen_vecs[0].y * eigen_val[0]));

	Point2f p2 = Point2f(static_cast<float>(eigen_vecs[1].x * eigen_val[1]), static_cast<float>(eigen_vecs[1].y * eigen_val[1]));
	Point2f p3 = (p1)+(p2);//Point2f(p1.x + p2.x, p1.y + p2.y);
	line(img, cntr, cntr + p1, Scalar(99, 99, 0), 1);
	line(img, cntr, cntr + p2, Scalar(99, 99, 0), 1);
	line(img, cntr, cntr + p3, Scalar(33, 33, 0), 1);

	return p3;
}

class Obj_Set
{
public:
	Obj_Set(size_t _n = 0, Mat _obj = Mat()):n(_n) , obj_img(_obj) {};

	~Obj_Set() {};

	size_t n;
	Mat obj_img;
	Point2f centre = Point2f(-1,-1);
	Point2f centre_diff = Point2f(0, 0);

	double master_dir = 0;
	double ang_diff = 0;
	double area = 1;

	Mat_<double> R;
	vector<Point2f> std_obj;
	Point2f master_vec;
	Rect2f rect;

	static Rect2f max_rect;
	static Size max_size;
};

Rect2f Obj_Set::max_rect = Rect2f();
Size Obj_Set::max_size = Size(0,0);

extern class Obj_Set;

vector<double> bin_search(Mat & obj_img, Mat & temp , double low_th , double high_th)
{
	Point2f obj_centre = Point2f(obj_img.cols / 2, obj_img.rows / 2);

	Mat tmp_img = temp;
	Mat blending;
	Mat_<double> diff;
	Mat obj_rot = Mat(obj_img.size(), obj_img.type());

	double IoU = 1;
	map<double, double> iou_set;
	double ang_gap = (high_th - low_th) / 2;
	double min_ang = 0;
	for (double ang = low_th; ang <= high_th; ang += ang_gap)
	{
		Mat_<double> R = getRotationMatrix2D(obj_centre, ang, 1);
		warpAffine(obj_img, obj_rot, R, obj_img.size(), INTER_CUBIC);
		//addWeighted(tmp_img, 0.7, obj_rot, 0.3, 0, blending);

		absdiff(temp, obj_rot, diff);
		double iou = sum(diff)[0] / sum(temp)[0];
		iou_set[iou] = ang;
	}
	vector<double> iou_2st;
	if (iou_set.size() < 2) return iou_2st;
	auto it1 = iou_set.begin();
	auto it2 = ++iou_set.begin();	
	iou_2st.push_back(it1->second);
	iou_2st.push_back(it2->second);
	return iou_2st;
}

/**
 *  \brief Automatic brightness and contrast optimization with optional histogram clipping
 *  \param [in]src Input image GRAY or BGR or BGRA
 *  \param [out]dst Destination image
 *  \param clipHistPercent cut wings of histogram at given percent tipical=>1, 0=>Disabled
 *  \note In case of BGRA image, we won't touch the transparency
*/
void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent = 0)
{

	CV_Assert(clipHistPercent >= 0);
	CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

	int histSize = 256;
	float alpha, beta;
	double minGray = 0, maxGray = 0;

	//to calculate grayscale histogram
	cv::Mat gray;
	if (src.type() == CV_8UC1) gray = src;
	else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
	else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
	if (clipHistPercent == 0)
	{
		// keep full available range
		cv::minMaxLoc(gray, &minGray, &maxGray);
	}
	else
	{
		cv::Mat hist; //the grayscale histogram

		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		// calculate cumulative distribution from the histogram
		vector<float> accumulator(histSize);
		accumulator[0] = hist.at<float>(0);
		for (int i = 1; i < histSize; i++)
		{
			accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
		}

		// locate points that cuts at required value
		float max = accumulator.back();
		clipHistPercent *= (max / 100.0); //make percent as absolute
		clipHistPercent /= 2.0; // left and right wings
		// locate left cut
		minGray = 0;
		while (accumulator[minGray] < clipHistPercent)
			minGray++;

		// locate right cut
		maxGray = histSize - 1;
		while (accumulator[maxGray] >= (max - clipHistPercent))
			maxGray--;
	}

	// current range
	float inputRange = maxGray - minGray;

	alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
	beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

	// Apply brightness and contrast normalization
	// convertTo operates with saurate_cast
	src.convertTo(dst, -1, alpha, beta);

	// restore alpha channel from source 
	if (dst.type() == CV_8UC4)
	{
		int from_to[] = { 3, 3 };
		cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
	}
	return;
}

void matrix_partitions(cv::Mat & _src, multimap<double, Rect2f, greater<double>> & indesity , Size sub_img_size, int stride , Mat & std)
{
	auto src_type = _src.type();
	_src.convertTo(_src, CV_8UC1);
	Mat src = _src;
	//threshold(src, src, 0, 1, THRESH_BINARY);
	//blur(src, src, Size(3, 3), Point(-1, -1), BORDER_ISOLATED);
	Mat c_map = Mat::zeros(src.size(), CV_32FC1);
	vector<Point2f>corners;
	double qualityLevel = 0.1;//角点检测可接受的最小特征
	double minDistance = max(sub_img_size.height, sub_img_size.width) + stride;//角点之间的最小距离
	int blockSize = min(sub_img_size.height, sub_img_size.width)/4;//计算导数自相关矩阵时指定的邻域范围

	goodFeaturesToTrack(src, corners, max(sub_img_size.height, sub_img_size.width) * stride, qualityLevel, minDistance , Mat() , blockSize ,true);
	_src.convertTo(_src, src_type);

	for (auto it = corners.begin(); it != corners.end(); ++it)
	{
		c_map(Rect2f(*it - Point2f(blockSize / 2, blockSize / 2), Size(blockSize, blockSize))) = 1.0 / (blockSize * blockSize);
	}
	draw_circle(src , corners);
	int borden = sqrt(sub_img_size.width * sub_img_size.width + sub_img_size.height * sub_img_size.height);
	for (int i = 1; i < src.rows - borden-1; i += stride)
	{
		for (int j = 1; j < src.cols - borden-1; j += stride)
		{
			Rect2f x_rext = Rect2f(Point2f(j, i), Point2f(j + sub_img_size.width, i + sub_img_size.height));
			
			Mat sub = std(x_rext);		
			double area = sum(sub)[0] / 255.0;
			if (area <= ( min(sub_img_size.height , sub_img_size.width) * 1.2 )) continue;

			float s_roi = 0.2;
			Point2f sub_br = Point2f(sub.cols, sub.rows) / 2;
			Rect2f t(sub_br * (1- s_roi), sub_br * (1 + s_roi));
			
			auto x = sum(sub(t))[0] / (t.area() * 255);
			area += sum( sub(t) )[0] / (t .area() * 255);
			double ws = sum(c_map(x_rext))[0];
			if (ws > 0)
			{
				indesity.insert(pair<double, Rect2f>(area / (1+ws), x_rext));
				continue;
			}
			//rectangle(sub, t, Scalar(233), 1, LINE_4);
			indesity.insert(pair<double, Rect2f>(area , x_rext));
		}
	}
	for (auto & des : indesity)
	{
		rectangle(std, des.second, Scalar(180), 1, LINE_4);
		break;
	}
}


void cvHilditchThin1(cv::Mat& src, cv::Mat& dst)
{
	//算法有问题，得不到想要的效果
	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	int i, j;
	int width, height;
	//之所以减2，是方便处理8邻域，防止越界
	width = src.cols - 2;
	height = src.rows - 2;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	int A1;
	cv::Mat tmpimg;
	while (1)
	{
		dst.copyTo(tmpimg);
		ifEnd = false;
		img = tmpimg.data + step;
		for (i = 2; i < height; i++)
		{
			img += step;
			for (j = 2; j < width; j++)
			{
				uchar* p = img + j;
				A1 = 0;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1] > 0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1] > 0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1] > 0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step] > 0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1] > 0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1] > 0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1] > 0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step] > 0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step] > 0 ? 1 : 0;
					p3 = p[-step + 1] > 0 ? 1 : 0;
					p4 = p[1] > 0 ? 1 : 0;
					p5 = p[step + 1] > 0 ? 1 : 0;
					p6 = p[step] > 0 ? 1 : 0;
					p7 = p[step - 1] > 0 ? 1 : 0;
					p8 = p[-1] > 0 ? 1 : 0;
					p9 = p[-step - 1] > 0 ? 1 : 0;
					//计算AP2,AP4
					int A2, A4;
					A2 = 0;
					//if(p[-step]>0)
					{
						if (p[-2 * step] == 0 && p[-2 * step + 1] > 0) A2++;
						if (p[-2 * step + 1] == 0 && p[-step + 1] > 0) A2++;
						if (p[-step + 1] == 0 && p[1] > 0) A2++;
						if (p[1] == 0 && p[0] > 0) A2++;
						if (p[0] == 0 && p[-1] > 0) A2++;
						if (p[-1] == 0 && p[-step - 1] > 0) A2++;
						if (p[-step - 1] == 0 && p[-2 * step - 1] > 0) A2++;
						if (p[-2 * step - 1] == 0 && p[-2 * step] > 0) A2++;
					}


					A4 = 0;
					//if(p[1]>0)
					{
						if (p[-step + 1] == 0 && p[-step + 2] > 0) A4++;
						if (p[-step + 2] == 0 && p[2] > 0) A4++;
						if (p[2] == 0 && p[step + 2] > 0) A4++;
						if (p[step + 2] == 0 && p[step + 1] > 0) A4++;
						if (p[step + 1] == 0 && p[step] > 0) A4++;
						if (p[step] == 0 && p[0] > 0) A4++;
						if (p[0] == 0 && p[-step] > 0) A4++;
						if (p[-step] == 0 && p[-step + 1] > 0) A4++;
					}


					//printf("p2=%d p3=%d p4=%d p5=%d p6=%d p7=%d p8=%d p9=%d\n", p2, p3, p4, p5, p6,p7, p8, p9);
					//printf("A1=%d A2=%d A4=%d\n", A1, A2, A4);
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7 && A1 == 1)
					{
						if (((p2 == 0 || p4 == 0 || p8 == 0) || A2 != 1) && ((p2 == 0 || p4 == 0 || p6 == 0) || A4 != 1))
						{
							dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							ifEnd = true;
							//printf("\n");

							//PrintMat(dst);
						}
					}
				}
			}
		}
		//printf("\n");
		//PrintMat(dst);
		//PrintMat(dst);
		//已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}
}

cv::Mat thinImage( cv::Mat & _src,int maxIterations = -1)
{
	Mat src = _src.clone();
	src.convertTo(src, CV_8UC1);
	threshold(src, src, 0, 1, THRESH_OTSU);
	
	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达
			break;
		vector<uchar *> mFlag; //用于标记需要删除的点
		//对点标记
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记
				//  p9 p2 p3
				//  p8 p1 p4
				//  p7 p6 p5
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 == 0)
					{
						//标记
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除
		for (vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空
		}

		//对点标记
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记
				//  p9 p2 p3
				//  p8 p1 p4
				//  p7 p6 p5
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//标记
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除
		for (vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空
		}
	}
	for (int i = 1; i < dst.rows - 1; ++i)
	{
		for (int j = 1; j < dst.cols - 1; ++j)
		{
			if (dst.at<uchar>(i, j) <= 0) continue;
			if (dst.at<uchar>(i - 1, j - 1) > 0)
			{
				dst.at<uchar>(i - 1, j) = 0;
				dst.at<uchar>(i, j - 1) = 0;
			}
			if (dst.at<uchar>(i + 1, j + 1) > 0)
			{
				dst.at<uchar>(i + 1, j) = 0;
				dst.at<uchar>(i, j + 1) = 0;
			}
			if (dst.at<uchar>(i + 1, j - 1) > 0)
			{
				dst.at<uchar>(i, j - 1) = 0;
				dst.at<uchar>(i + 1, j) = 0;
			}
			if (dst.at<uchar>(i - 1, j + 1) > 0)
			{
				dst.at<uchar>(i - 1, j) = 0;
				dst.at<uchar>(i, j + 1) = 0;
			}
		}
	}

	return dst;
}

class diff_map
{
public:
	size_t n;
	Mat diff;
	double iou;
	double max_scale_diff;
	double angle_diff;
	Point2f obj_center_in_frame;
	diff_map(size_t _n = 0 , Mat _diff = Mat() , double _iou  = 0, double _max_s = 0, vector<Point2f> tripple_obj = vector<Point2f>())
	{
		n = _n;
		diff = _diff;
		iou = _iou;
		max_scale_diff = _max_s;
	};
	~diff_map() {};

};

vector<int> GetFlags(int a[], int length)
{
	vector<int> vec;
	int neighbour[] = { 1,2,4,8,16,32,64,128,1,2,4,8,16,32,64 };
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			int sum = 0;
			for (int k = j; k <= j + a[i]; k++)
				sum += neighbour[k];
			vec.push_back(sum);
		}
	}
	return vec;
}
Mat skeleton(cv::Mat & _Input) //Input-binary image
{
	Mat Input = _Input.clone();
	Input.convertTo(Input, CV_8UC1);
	threshold(Input, Input, 0, 1, THRESH_OTSU);
	int a0[] = { 1,2,3,4,5,6 };
	int a1[] = { 2 };
	int a2[] = { 2,3 };
	int a3[] = { 2,3,4 };
	int a4[] = { 2,3,4,5 };
	int a5[] = { 2,3,4,5,6 };
	vector<int> A0 = GetFlags(a0, 6);

	vector<int> A1 = GetFlags(a1, 1);

	vector<int> A2 = GetFlags(a2, 2);
	vector<int> A3 = GetFlags(a3, 3);
	vector<int> A4 = GetFlags(a4, 4);
	vector<int> A5 = GetFlags(a5, 5);
	vector<cv::Point2i> border;
	bool modify = true;
	int neighbour[3][3] = {
		{128,1,2},
		{64,0,4},
		{32,16,8}
	};
	int row = Input.rows;
	int col = Input.cols;
	while (modify)
	{
		modify = false;
		// flag the border Pharse 0
		for (int m = 1; m < row - 1; ++m)
		{
			for (int n = 1; n < col - 1; ++n)
			{
				int weight = 0;
				for (int j = -1; j <= 1; ++j)
				{
					for (int k = -1; k <= 1; k++)
					{
						weight += neighbour[j + 1][k + 1] * Input.at<uchar>(m + j, n + k);
					}
				}
				if (find(A0.begin(), A0.end(), weight) != A0.end())
					border.push_back(cv::Point2i(m, n));
			}
		}
		//Pharse 1
		vector<cv::Point2i>::iterator first = border.begin();
		while (first != border.end())
		{
			int weight = 0;
			for (int j = -1; j <= 1; ++j)
			{
				for (int k = -1; k <= 1; k++)
				{
					weight += neighbour[j + 1][k + 1] * Input.at<uchar>((*first).x + j, (*first).y + k);
				}
			}
			if (find(A1.begin(), A1.end(), weight) != A1.end())
			{
				Input.at<uchar>((*first).x, (*first).y) = 0;
				first = border.erase(first);
			}
			else
				++first;
		}
		//Pharse2
		first = border.begin();
		while (first != border.end())
		{
			int weight = 0;
			for (int j = -1; j <= 1; ++j)
			{
				for (int k = -1; k <= 1; k++)
				{
					weight += neighbour[j + 1][k + 1] * Input.at<uchar>((*first).x + j, (*first).y + k);
				}
			}
			if (find(A2.begin(), A2.end(), weight) != A2.end())
			{
				Input.at<uchar>((*first).x, (*first).y) = 0;
				first = border.erase(first);
			}
			else
				++first;
		}
		//Pharse3
		first = border.begin();
		while (first != border.end())
		{
			int weight = 0;
			for (int j = -1; j <= 1; ++j)
			{
				for (int k = -1; k <= 1; k++)
				{
					weight += neighbour[j + 1][k + 1] * Input.at<uchar>((*first).x + j, (*first).y + k);
				}
			}
			if (find(A3.begin(), A3.end(), weight) != A3.end())
			{
				Input.at<uchar>((*first).x, (*first).y) = 0;
				first = border.erase(first);
			}
			else
				++first;
		}
		//Pharse4
		first = border.begin();
		while (first != border.end())
		{
			int weight = 0;
			for (int j = -1; j <= 1; ++j)
			{
				for (int k = -1; k <= 1; k++)
				{
					weight += neighbour[j + 1][k + 1] * Input.at<uchar>((*first).x + j, (*first).y + k);
				}
			}
			if (find(A4.begin(), A4.end(), weight) != A4.end())
			{
				Input.at<uchar>((*first).x, (*first).y) = 0;
				first = border.erase(first);
			}
			else
				++first;
		}
		//Pharse5
		first = border.begin();
		while (first != border.end())
		{
			int weight = 0;
			for (int j = -1; j <= 1; ++j)
			{
				for (int k = -1; k <= 1; k++)
				{
					weight += neighbour[j + 1][k + 1] * Input.at<uchar>((*first).x + j, (*first).y + k);
				}
			}
			if (find(A5.begin(), A5.end(), weight) != A5.end())
			{
				Input.at<uchar>((*first).x, (*first).y) = 0;
				first = border.erase(first);
				modify = true;
			}
			else
				++first;
		}
		//Pharse6
		border.clear();
	}
	for (int m = 1; m < row - 1; ++m)
	{
		for (int n = 1; n < col - 1; ++n)
		{
			int weight = 0;
			for (int j = -1; j <= 1; ++j)
			{
				for (int k = -1; k <= 1; k++)
				{
					weight += neighbour[j + 1][k + 1] * Input.at<uchar>(m + j, n + k);
				}
			}
			if (find(A0.begin(), A0.end(), weight) != A0.end())
				Input.at<uchar>(m, n) = 0;;
		}
	}

	for (int i = 1; i < Input.rows - 1; ++i)
	{
		for (int j = 1; j < Input.cols - 1; ++j)
		{
			if (Input.at<uchar>(i, j) <= 0) continue;
			if (Input.at<uchar>(i - 1, j - 1) > 0)
			{
				Input.at<uchar>(i - 1, j) = 0;
				Input.at<uchar>(i, j - 1) = 0;
			}
			if (Input.at<uchar>(i + 1, j + 1) > 0)
			{
				Input.at<uchar>(i + 1, j) = 0;
				Input.at<uchar>(i, j + 1) = 0;
			}
			if (Input.at<uchar>(i + 1, j - 1) > 0)
			{
				Input.at<uchar>(i, j - 1) = 0;
				Input.at<uchar>(i + 1, j) = 0;
			}
			if (Input.at<uchar>(i - 1, j + 1) > 0)
			{
				Input.at<uchar>(i - 1, j) = 0;
				Input.at<uchar>(i, j + 1) = 0;
			}
		}
	}
	return Input;
}

set<int> GetAi(int a[], int length)//获取A0~A5
{
	set<int> vec;
	int neighbour[] = { 1,2,4,8,16,32,64,128,1,2,4,8,16,32,64 };
	for (int i = 0; i < length; i++)
		for (int j = 0; j < 8; j++)
		{
			int sum = 0;
			for (int k = j; k <= j + a[i]; k++)
				sum += neighbour[k];
			vec.insert(sum);
		}
	return vec;
}
//迭代腐蚀
bool erodephase(list<cv::Point> &border, cv::Mat&Input, int neighbour[][3], const set<int>& A)
{
	auto pt = border.begin();
	bool result = false;
	while (pt != border.end())
	{

		int weight = 0;
		for (int j = -1; j <= 1; ++j)
			for (int k = -1; k <= 1; k++)
				weight += neighbour[j + 1][k + 1] * Input.at<uchar>(pt->y + j, pt->x + k);

		if (find(A.begin(), A.end(), weight) != A.end())
		{
			Input.at<uchar>(pt->y, pt->x) = 0;
			pt = border.erase(pt);
			result = true;
		}
		else
			++pt;
	}
	return result;
}
//找边界 
void findborder(list<cv::Point2i>& border, const cv::Mat&Input)
{
	int cnt = 0;
	int rows = Input.rows;
	int cols = Input.cols;
	cv::Mat bordermat = Input.clone();
	for (int row = 1; row < rows - 1; ++row)
		for (int col = 1; col < cols - 1; ++col)
		{
			int weight = 0;
			for (int j = -1; j <= 1; ++j)
				for (int k = -1; k <= 1; k++)
				{
					if (Input.at<uchar>(row + j, col + k) == 1)
						++cnt;
				}
			if (cnt == 9)
				bordermat.at<uchar>(row, col) = 0;
			cnt = 0;
		}

	for (int row = 1; row < rows - 1; ++row)
		for (int col = 1; col < cols - 1; ++col)
		{
			if (bordermat.at<uchar>(row, col) == 1)
				border.push_back(cv::Point2i(col, row));
		}

}
//最后一步，得到骨架
void finalerode(cv::Mat&Input, int neighbour[][3], const set<int>& A)
{
	int rows = Input.rows;
	int cols = Input.cols;
	for (int m = 1; m < rows - 1; ++m)
		for (int n = 1; n < cols - 1; ++n)
		{
			int weight = 0;
			for (int j = -1; j <= 1; ++j)
				for (int k = -1; k <= 1; k++)
				{
					weight += neighbour[j + 1][k + 1] * Input.at<uchar>(m + j, n + k);
				}

			if (find(A.begin(), A.end(), weight) != A.end())
				Input.at<uchar>(m, n) = 0;
		}
}
Mat thin_3(cv::Mat & _Input) //Input是二值图像
{
	Mat Input = _Input.clone();
	blur(Input, Input, Size(3, 3), Point(-1, -1), BORDER_ISOLATED);
	Input.convertTo(Input, CV_8UC1);
	threshold(Input, Input, 0, 1, THRESH_OTSU);
	
	int a0[] = { 1,2,3,4,5,6 ,7};
	int a[] = { 1 };
	int a1[] = { 2 };
	int a2[] = { 2,3 };
	int a3[] = { 2,3,4 };
	int a4[] = { 2,3,4,5 };
	int a5[] = { 2,3,4,5,6 };
	int a6[] = { 2,3,4,5,6,7};
	set<int> A0 = GetAi(a0, 6);
	set<int> A = GetAi(a, 0);
	set<int> A1 = GetAi(a1, 1);
	set<int> A2 = GetAi(a2, 2);
	set<int> A3 = GetAi(a3, 3);
	set<int> A4 = GetAi(a4, 4);
	set<int> A5 = GetAi(a5, 5);
	list<cv::Point2i> border;
	bool continue_ = true;
	int neighbour[3][3] = {
		{ 128,1,2 },
		{ 64,0,4 },
		{ 32,16,8 }
	};
	while (continue_)
	{
		double area = sum(Input)[0];
		findborder(border, Input);//Phase0
		//可以在下面每一步打印结果，看每一步对提取骨架的贡献
		//finalerode(Input, neighbour, A7);
		//erodephase(border, Input, neighbour, A);//Phase1
		erodephase(border, Input, neighbour, A1);//Phase1
		erodephase(border, Input, neighbour, A2);//Phase2
		erodephase(border, Input, neighbour, A3);//Phase3
		erodephase(border, Input, neighbour, A4);//Phase4
		erodephase(border, Input, neighbour, A5);//Phase5
		border.clear();
		continue_ = (fabs(area - sum(Input)[0]) / area) > 0.05;
		//blur(Input, Input, Size(1, 1), Point(-1, -1), BORDER_ISOLATED);
	}
	//finalerode(Input, neighbour, A0);//最后一步

	for (int i = 1; i < Input.rows - 1; ++i)
	{
		for (int j = 1; j < Input.cols - 1; ++j)
		{
			if (Input.at<uchar>(i, j) <= 0) continue;
			if (Input.at<uchar>(i - 1, j - 1) > 0)
			{
				Input.at<uchar>(i - 1, j) = 0;
				Input.at<uchar>(i, j-1) = 0;
			}
			if (Input.at<uchar>(i + 1, j + 1) > 0)
			{
				Input.at<uchar>(i + 1, j) = 0;
				Input.at<uchar>(i, j + 1) = 0;
			}
			if (Input.at<uchar>(i + 1, j - 1) > 0)
			{
				Input.at<uchar>(i, j - 1) = 0;
				Input.at<uchar>(i + 1, j ) = 0;
			}
			if (Input.at<uchar>(i - 1, j + 1) > 0)
			{
				Input.at<uchar>(i - 1, j) = 0;
				Input.at<uchar>(i, j + 1) = 0;
			}
		}
	}
	return Input;
}


void thinImage2(Mat & srcImg) 
{
	threshold(srcImg, srcImg, 127, 1, THRESH_BINARY);
	srcImg.convertTo(srcImg, CV_8UC1);
	vector<Point> deleteList;
	int neighbourhood[9];
	int nl = srcImg.rows;
	int nc = srcImg.cols;
	bool inOddIterations = true;
	while (true) {
		for (int j = 1; j < (nl - 1); j++) {
			uchar* data_last = srcImg.ptr<uchar>(j - 1);
			uchar* data = srcImg.ptr<uchar>(j);
			uchar* data_next = srcImg.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++) {
				if (data[i] == 255) {
					int whitePointCount = 0;
					neighbourhood[0] = 1;
					if (data_last[i] == 255) neighbourhood[1] = 1;
					else  neighbourhood[1] = 0;
					if (data_last[i + 1] == 255) neighbourhood[2] = 1;
					else  neighbourhood[2] = 0;
					if (data[i + 1] == 255) neighbourhood[3] = 1;
					else  neighbourhood[3] = 0;
					if (data_next[i + 1] == 255) neighbourhood[4] = 1;
					else  neighbourhood[4] = 0;
					if (data_next[i] == 255) neighbourhood[5] = 1;
					else  neighbourhood[5] = 0;
					if (data_next[i - 1] == 255) neighbourhood[6] = 1;
					else  neighbourhood[6] = 0;
					if (data[i - 1] == 255) neighbourhood[7] = 1;
					else  neighbourhood[7] = 0;
					if (data_last[i - 1] == 255) neighbourhood[8] = 1;
					else  neighbourhood[8] = 0;
					for (int k = 1; k < 9; k++) {
						whitePointCount += neighbourhood[k];
					}
					if ((whitePointCount >= 2) && (whitePointCount <= 6)) {
						int ap = 0;
						if ((neighbourhood[1] == 0) && (neighbourhood[2] == 1)) ap++;
						if ((neighbourhood[2] == 0) && (neighbourhood[3] == 1)) ap++;
						if ((neighbourhood[3] == 0) && (neighbourhood[4] == 1)) ap++;
						if ((neighbourhood[4] == 0) && (neighbourhood[5] == 1)) ap++;
						if ((neighbourhood[5] == 0) && (neighbourhood[6] == 1)) ap++;
						if ((neighbourhood[6] == 0) && (neighbourhood[7] == 1)) ap++;
						if ((neighbourhood[7] == 0) && (neighbourhood[8] == 1)) ap++;
						if ((neighbourhood[8] == 0) && (neighbourhood[1] == 1)) ap++;
						if (ap == 1) {
							if (inOddIterations && (neighbourhood[3] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[5] == 0)) {
								deleteList.push_back(Point(i, j));
							}
							else if (!inOddIterations && (neighbourhood[1] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[7] == 0)) {
								deleteList.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deleteList.size() == 0)
			break;
		for (size_t i = 0; i < deleteList.size(); i++) {
			Point tem;
			tem = deleteList[i];
			uchar* data = srcImg.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deleteList.clear();

		inOddIterations = !inOddIterations;
	}
}

void img_range_remapping(cv::Mat & mat)
{
	mat.convertTo(mat, CV_64FC1);
	double max = 0, min = 0;
	cv::Point minPoint, maxPoint;
	cv::minMaxLoc(mat, &min, &max, &minPoint, &maxPoint);
	mat -= min;
	mat = (mat / (max - min) ) * 255.0;
}

vector<Rect2d> matching_template_rois(Mat & _img, Mat & _patch , double scaler_f = 0.2 , double similarity = 0.6, int k = 32, double roi_scaler = 0.2 ,double roi_expend = 0.05 )
{
	Mat img, patch;
	cv::resize(_img, img, Size(), scaler_f, scaler_f, INTER_AREA);
	cv::resize(_patch, patch, Size(), scaler_f, scaler_f, INTER_AREA);

	Mat fus_map;
	patch = patch(Rect2d(0, 0, patch.cols, patch.rows) & Rect2d(0, 0, img.cols, img.rows));
	matchTemplate(img, patch, fus_map, TM_CCOEFF_NORMED);
 	threshold(fus_map, fus_map, similarity, 255, THRESH_TOZERO);

	multimap<float, Rect2d ,greater<float> >obj_st;
	for (size_t i = 0; i < k; ++i)
	{
		double maxVal; Point maxLoc;
		minMaxLoc(fus_map, NULL, &maxVal, NULL, &maxLoc);

		if (maxVal <= 0) break;
		Point2f matchLoc = maxLoc;
		Rect2d roi = Rect2f(matchLoc, Size(patch.cols, patch.rows));

		Point2d scale_shift = Point2d(roi.width, roi.height) * roi_scaler;
		fus_map( Rect2d(roi.tl() - scale_shift, roi.br() + scale_shift) & Rect2d( 0, 0 , fus_map.cols, fus_map.rows) ) = 0;
		roi = Rect2d( (roi.tl() / scaler_f - Point2d(_patch.cols, _patch.rows) * roi_expend) , Size(_patch.cols * (1+ 2*roi_expend) , _patch.rows * (1+ 2*roi_expend))) & Rect2d(0, 0, _img.cols, _img.rows);
		//rectangle(_img, roi, Scalar(222), 1, LINE_4);
		obj_st.insert(pair<float, Rect2d>(roi.tl().x, roi));
	}

	vector<Rect2d> obj_tar;
	for (auto & _obj : obj_st)
	{
		obj_tar.push_back(_obj.second);
	}
	return obj_tar;
}

vector<Point2f> master_vec_to_tripple(Point2f cen, Point2f master_vec)
{
	vector<Point2f> src_trip(3);
	src_trip[0] = cen;
	src_trip[1] = master_vec / norm(master_vec) * ((cen.x + cen.y) / 2.0) + cen;

	Mat X = (Mat_<double>(3, 1) << src_trip[1].x, src_trip[1].y, 1);
	X = (getRotationMatrix2D(src_trip[0], -90, 1) * X);
	src_trip[2] = Point2f(X.at<double>(Point(0, 0)), X.at<double>(Point(0, 1)));

	Point2f v1 = src_trip[1] - src_trip[0];
	Point2f v2 = src_trip[2] - src_trip[0];
	double  ang = acos(v1.dot(v2) / (norm(v1) * norm(v2))) * 180 / 3.1415926;

	if (fabs(ang - 90) > 1)
	{
		cout << "error : tripple master point is not orthogonal" << endl;
	}
	return src_trip;
}



void matching_template_accelerated( vector<Mat> & objs ,Mat & std_temp ,vector<diff_map> & diff_set , double area_th = 0.01)
{
		int i = 0;
		multimap<float, Obj_Set > obj_st;
		for (auto & obj : objs)
		{
			obj_st.insert(pair<float, Obj_Set>(i, Obj_Set(i, obj.clone())));
			++i;
		}
		vector<vector<Point>> contours;
		double avg_ang = 0;

		for (auto & _obj : obj_st)
		{
			Obj_Set & obj = _obj.second;
			Mat & obj_img = obj.obj_img;
			
			cv::findContours(obj_img, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			Point2f master_vec = Point2f(0, 0);
			vector<Point2f> avg_cen;
			for (auto & c : contours)
			{
				double area_rate = moments(c, true).m00 / (obj_img.cols * obj_img.rows);
				if (area_rate < area_th)
				{
					obj_img(boundingRect(c)) = 0;
					continue;
				}
				Moments Mpq = moments(c, true);
				Point2f centre = Point2f(static_cast<float>(Mpq.m10 / Mpq.m00), static_cast<float>(Mpq.m01 / Mpq.m00));
				
				avg_cen.push_back(centre);
				master_vec += getOrientation(c, obj_img, centre);
				obj.std_obj.insert(obj.std_obj.end(), c.begin(), c.end());
			}
			if (avg_cen.empty()) continue;
			Point2f cen = avg_cen[0];
			for (size_t i = 1 ; i < avg_cen.size() ; ++i)
			{
				cen = (avg_cen[i] + cen) / 2.0;
			}
			//circle(obj_img, cen, 3, Scalar(122, 122, 122), 1);

			//Moments Mpq = moments(obj.std_obj, true);
			//obj.centre = Point2f(static_cast<float>(Mpq.m10 / Mpq.m00), static_cast<float>(Mpq.m01 / Mpq.m00));
			//circle(obj_img, obj.centre, 3, Scalar(122, 122, 122), 1);
			obj.centre = cen;
			double master_dir = atan2(master_vec.y, master_vec.x) * 180 / 3.1415926;
			//master_dir = master_dir > 0 ? master_dir : 360 + master_dir;
			avg_ang += master_dir;

			obj.master_dir = master_dir;
			obj.master_vec = master_vec;
		}
		avg_ang /= (double)obj_st.size();
		Rect2f max_rect;


		Point2f center_max = Point2f(0, 0);
		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			obj.ang_diff = os.second.master_dir - avg_ang;
			Mat_<double> R = getRotationMatrix2D(obj.centre, obj.ang_diff, 1);

			vector<Point2f> std_obj = obj.std_obj;
			Rect2f rect = boundingRect(std_obj);
			//rectangle(obj_img, rect, Scalar(222), 1, LINE_4);
			Point2f vertices[4];
			vertices[0] = rect.tl(); vertices[1] = Point2f(rect.br().x, rect.tl().y); vertices[2] = rect.br(); vertices[3] = Point2f(rect.tl().x, rect.br().y);
			//for (int i = 0; i < 4; i++)
				//line(obj_img, vertices[i], vertices[(i + 1) % 4], Scalar(244), 1, LINE_4);
			//circle(obj_img, obj.centre, 3, Scalar(122, 122, 122), 1);

			double neg4_x = 0, neg4_y = 0;
			double max4_x = 0, max4_y = 0;
			for (int i = 0; i < 4; i++)
			{
				Mat X = (Mat_<double>(3, 1) << vertices[i].x, vertices[i].y, 1);
				X = (R * X);
				Point2f v = Point2f(X.at<double>(Point(0, 0)), X.at<double>(Point(0, 1)));
				if (v.x < neg4_x) neg4_x = v.x;
				if (v.y < neg4_y) neg4_y = v.y;

				if (v.x > max4_x) max4_x = v.x;
				if (v.y > max4_y) max4_y = v.y;
			}

			R.at<double>(Point(2, 0)) -= neg4_x;
			R.at<double>(Point(2, 1)) -= neg4_y; 
			obj.R = R;
			obj.centre -= Point2f(neg4_x, neg4_y);

			if (center_max.x < obj.centre.x) center_max.x = obj.centre.x;
			if (center_max.y < obj.centre.y) center_max.y = obj.centre.y;

			max4_x -= neg4_x;
			max4_y -= neg4_y;

			if (obj.max_size.width < max4_x) obj.max_size.width = max4_x;
			if (obj.max_size.height < max4_y) obj.max_size.height = max4_y;
		}

		Point2f max_cen_diff = Point2f(0,0);
		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			obj.centre_diff = center_max - obj.centre;
			if (obj.centre_diff.x > max_cen_diff.x) max_cen_diff.x = obj.centre_diff.x;
			if (obj.centre_diff.y > max_cen_diff.y) max_cen_diff.y = obj.centre_diff.y;
		}
		Obj_Set::max_size.width += max_cen_diff.x;
		Obj_Set::max_size.height += max_cen_diff.y;

		//Mat temp;
		//vector<Mat> obj_set;
		//Mat xx;
		
		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			Mat & obj_img = obj.obj_img;

			obj.R.at<double>(Point(2, 0)) += obj.centre_diff.x;
			obj.R.at<double>(Point(2, 1)) += obj.centre_diff.y;

			obj.centre += obj.centre_diff;
			warpAffine(obj_img, obj_img, obj.R, obj.max_size, INTER_AREA);
			//circle(obj.obj_img, obj.centre, 3, Scalar(155, 155, 155), 1);

			double len = sqrt(pow(obj_img.rows, 2) + pow(obj_img.cols, 2));
			Mat_<double> std_temp = Mat_<double>(Size(len, len), 0);

			Point2f temp_centre = Point2f(std_temp.cols / 2.0, std_temp.rows / 2.0);
			Point2f tl = temp_centre - obj.centre;
			Rect2f roi = Rect(tl, obj_img.size());

			//rectangle(std_temp, roi, Scalar(222), 1, LINE_4);
			obj_img.copyTo(std_temp(roi));
			obj.obj_img = std_temp;
			//circle(std_temp, temp_centre, 3, Scalar(222, 222, 222), 1);
			//blur(obj_img, obj_img, Size(1, 1), Point(-1, -1), BORDER_ISOLATED);
			//obj_set.push_back(obj_img);

			//xx.push_back(std_temp.t());
		}
		//xx = xx.t();
		//imwrite("obj4.jpg", xx);
		Mat temp = obj_st.find(0)->second.obj_img;
		obj_st.erase(0);
		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			Mat & obj_img = obj.obj_img;
			Mat tmp_img = temp;

			double th_up = 30;
			double th_down = -30;
			double err_gap = 0.5;
			double closer = fabs(fabs(th_up - th_down) - err_gap);
			int abnormal_times = 10;
			for (; fabs(th_up - th_down) >= err_gap *(0.1*abnormal_times) ;)
			{
				vector<double> new_th = bin_search(obj_img, tmp_img, th_down, th_up);
				if (new_th.size() <= 1) break;
				th_down = min(new_th[0], new_th[1]);
				th_up = max(new_th[0], new_th[1]);
				double s = fabs(fabs(th_up - th_down) - err_gap);
				if (fabs(fabs(th_up - th_down) - err_gap) >= closer) ++abnormal_times;
				//if (abnormal_times > 10) break;
				closer = fabs(fabs(th_up - th_down) - err_gap);
			}
			obj.ang_diff = (th_down + th_up) / 2.0;
		}

		vector<pair<double,double>> iou_set;
		size_t n = 0;
		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			Mat & obj_img = obj.obj_img;
			Mat tmp_img = temp;
			Mat blending ,diff;

			Point2f obj_centre = Point2f(obj_img.cols / 2, obj_img.rows / 2);
			Mat_<double> R = getRotationMatrix2D(obj_centre, obj.ang_diff, 1);
			warpAffine(obj_img, obj_img, R, obj_img.size(), INTER_AREA);
#ifdef _DEBUG
			addWeighted(tmp_img, 0.7, obj_img, 0.3, 0, blending);
#endif
			absdiff(temp, obj_img, diff);
			double iou = sum(diff)[0] / sum(temp)[0];

			multimap<double, Rect2f, greater<double>> indesity;
			int s = diff.cols / 20;
			Size S(s, s);
			int stride = min(S.height, S.width) / 4;
			matrix_partitions(diff, indesity, S, stride , temp);
			double max_width = 0;
			Mat sub, thin;
			for (auto & v : indesity)
			{
				sub = diff(v.second).clone();
				Mat sub_s(Size(sub.size().width * 1.2 , sub.size().height * 1.2), sub.type(), Scalar(0));
				Point2f tl = Point2f((sub_s.cols - sub.cols) / 2.0, (sub_s.rows - sub.rows) / 2.0);
				Rect2f roi = Rect(tl, sub.size());

				sub.copyTo(sub_s(roi));
#ifdef _DEBUG 
				Mat thin1 = thin_3(sub_s);		
#endif
				Mat thin = thin_3(sub_s);
				//Mat thin_s = thin_3(sub_s);
				double area = sum(sub_s)[0] / 255;
				double len = sum(thin)[0];

				max_width = area / len ;
				rectangle(diff, v.second, Scalar(244), 2, LINE_4);
				if (max_width >= min(S.height , S.width))
				{
					sub_s = sub_s.t();
					
					thin = skeleton(sub_s);
					len = sum(thin)[0];
					max_width = area / len;
				}
				if (max_width >= min(S.height, S.width)) max_width = min(S.height, S.width);
				//max_width /= scaler_f2;
				break;
			}

			diff_set.push_back( diff_map(n++, diff, iou, max_width) );
		}
		Obj_Set::max_rect = Rect2f();
		Obj_Set::max_size = Size(0, 0);
		return ;
}

			/*
			{
				vector<vector<Point>> contours;
				findContours(obj_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

				vector<Point2f> std_obj, temp;
				Point2f master_vec = Point2f(0, 0);
				vector<Point2f> avg_center;
				double area_all = obj_img.cols * obj_img.rows;
				for (auto & c : contours)
				{
					double area_rate = moments(c, true).m00 / area_all;
					if (area_rate < area_th || area_rate > 1 - area_th)
					{
						continue;
					}
					//centre_set.push_back(centre);
					Moments Mpq = moments(c, true);
					Point2f centre = Point2f(static_cast<float>(Mpq.m10 / Mpq.m00), static_cast<float>(Mpq.m01 / Mpq.m00));
					//avg_center.push_back(centre);
					circle(obj_img, centre, 2, Scalar(33, 33, 33), 1);
					master_vec += getOrientation2(c, obj_img, centre);

					std_obj.insert(std_obj.end(), c.begin(), c.end());
				}
				Moments Mpq = moments(std_obj, true);
				Point2f centre = Point2f(static_cast<float>(Mpq.m10 / Mpq.m00), static_cast<float>(Mpq.m01 / Mpq.m00));

				circle(obj_img, centre, 2, Scalar(33, 33, 33), 1);
				line(obj_img, centre, centre + master_vec, Scalar(33, 33, 0), 1);

				double master_dir = atan2(master_vec.y, master_vec.x) * 180 / 3.1415926;

			}
		
		{/*
			contours.clear();
			findContours(obj_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			double area_all = obj_img.cols * obj_img.rows;
			std_obj.clear();
			for (auto & c : contours)
			{
				double area_rate = moments(c, true).m00 / area_all;
				if (area_rate < area_th || area_rate > 1 - area_th)
				{
					continue;
				}
				std_obj.insert(std_obj.end(), c.begin(), c.end());
			}
			rect = boundingRect(std_obj);
			if (rect.height > max_rect.height ) max_rect.height = rect.height;
			if (rect.width > max_rect.width) max_rect.width = rect.width;
			rectangle(obj_img, rect, Scalar(222), 1, LINE_4);
			normed_obj_set[n++] = obj_img(rect);
		}

		map<size_t, double> iou_set;
		//n = 0;
		Mat_<double> std_temp = Mat_<double>(max_rect.size(), 0);
		Mat img_obj = normed_obj_set[1];
		double temp_area = sum(img_obj)[0];
		Point2f tl = Point2f((std_temp.cols - img_obj.cols) / 2.0, (std_temp.rows - img_obj.rows) / 2.0);
		Rect2f roi = Rect(tl, img_obj.size());
		img_obj.copyTo(std_temp(roi));

		for (auto & map_obj : normed_obj_set)
		{
			Mat_<double> std_obj = Mat_<double>(max_rect.size(), 0);
			Mat img_obj = map_obj.second;
			Point2f tl = Point2f( (std_obj.cols - img_obj.cols)/2.0 , (std_obj.rows - img_obj.rows)/2.0);
			Rect2f roi = Rect(tl, img_obj.size());
			img_obj.copyTo(std_obj(roi));

			Mat_<double> diff;
			absdiff(std_temp, std_obj , diff);
			double diff_area = sum(diff)[0];
			double IoU = diff_area / temp_area;

			iou_set[++n] = IoU;
			//double psnr = getPSNR(std_temp, std_obj);
			
		}
	}
	//addWeighted(obj, alpha, src2, beta, 0.0 , );

	
	Mat img, patch;
	cv::resize(_img, img, Size(), scaler_f, scaler_f, INTER_CUBIC);
	cv::resize(_patch, patch, Size(), scaler_f, scaler_f, INTER_CUBIC);

	vector<vector<Point>> contours;
	map<size_t, vector<Point>, less<float> > centre_map;
	map<size_t, vector<Point>, greater<float> > area_map;

	findCont                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           nours(patch, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	{
#ifdef _DEBUG
		vector<vector<Point>> contours_x;
		for (auto & cm : contours)
		{
			contours_x.clear();
			contours_x.push_back(cm);
			drawContours(patch, contours_x, -1, Scalar(120), FILLED);
		}
#endif
	}

	vector<Point> std_obj; // = area_map.begin()->second;
	float std_area; // = area_map.begin()->second;
	for (auto & c : contours)
	{
		area_map[moments(c, true).m00] = c;
	}
	std_area = area_map.begin()->first;
	std_obj = area_map.begin()->second;


	centre_map.clear();
	area_map.clear();
	findContours(img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	std_area = moments(contours[3], true).m00;
	std_obj = contours[3];

	double sim_area = 0.9, sim_scale= 0.9;
	for (auto & c : contours)
	{
		Moments Mpq = moments(c, true);

		float area = Mpq.m00;
		double a_sim = 1.0 - fabs(area - std_area) / area;
		if (a_sim < sim_area) continue;

		double s_sim = 1.0 - matchShapes(c , std_obj , CONTOURS_MATCH_I3, 0);
		if (s_sim < sim_scale) continue;

		Point2f centre = Point2f(static_cast<float>(Mpq.m10 / Mpq.m00), static_cast<float>(Mpq.m01 / Mpq.m00));
		centre_map[centre.x] = c;
	}
	
#ifdef _DEBUG
	vector<vector<Point>> contours_x;
	for (auto & cm : centre_map)
	{
		contours_x.clear();
		contours_x.push_back(cm.second);
		//drawContours(img, contours_max, -1, Scalar(120), FILLED);
	}
#endif
	map<size_t , RotatedRect> rot_rects;
	map<size_t, Mat > obj_set;
	size_t n = 0;
	for (auto & cm : centre_map)
	{
		RotatedRect r_rect = minAreaRect(cm.second);
		//rot_rects[++n] = img(r_rect.boundingRect);
		rot_rects[++n] = r_rect;

	    //Point2f vertices[4];      //定义矩形的4个顶点
		//r_rect.points(vertices);   //计算矩形的4个顶点
	    //for (int i = 0; i < 4; i++)
		//line(img, vertices[i], vertices[(i + 1) % 4], Scalar(255, 255, 0), 1 , LINE_AA);

		Rect rect = boundingRect(cm.second);
		//rectangle(img, rect, CV_RGB(0, 0, 188));
		obj_set[n] = img(rect);
		Mat X = obj_set[n];
		Point2f centre = Point2f( (rect.br().x + rect.tl().x)/2 - rect.tl().x, (rect.br().y + rect.tl().y) / 2 - rect.tl().y);
		auto s = max(obj_set[n].cols, obj_set[n].rows);
		warpAffine(obj_set[n], obj_set[n], getRotationMatrix2D(centre, r_rect.angle, 1), Size(s,s), INTER_CUBIC);
		Mat X2 = obj_set[n];
		threshold(X2, X2, 127, 255, THRESH_BINARY);
	}
	for (auto & o : obj_set)
	{
		Mat X = o.second;
		vector<vector<Point>> contours;
		findContours(o.second, contours, RETR_TREE, CHAIN_APPROX_TC89_L1);

		vector<Point> obj;
		for (auto & c : contours)
		{
			area_map[moments(c, true).m00] = c;
		}
		obj = area_map.begin()->second;

		RotatedRect r_rect = minAreaRect(obj);
	}

	return obj_set;
}*/

vector<Point2f> match_knn_img_patch_centre( Mat & _img,  Mat & _patch, int k = 3 , double similarity = 0.95 , double roi_scaler = 0.1 , double scaler_f = 0.1)
{
	Mat img, patch;
	cv::resize(_img, img, Size(), scaler_f, scaler_f, INTER_CUBIC);
	cv::resize(_patch, patch, Size(), scaler_f, scaler_f , INTER_CUBIC);
	Mat fus_map;
	vector<Point2f> src_trip;
	if (patch.rows > img.rows || patch.cols > img.cols)return src_trip;
	matchTemplate(img, patch, fus_map, TM_CCORR_NORMED);

	normalize(fus_map, fus_map, 0, 1, NORM_MINMAX, -1);
	threshold(fus_map, fus_map, similarity, 255, THRESH_TOZERO);

	vector<Rect2f> roi_mask_trip, roi_trip;
	for (int i = 0; i < k; ++i)
	{
		double maxVal; Point maxLoc;
		minMaxLoc(fus_map, NULL, &maxVal, NULL, &maxLoc);
		if (maxVal <= 0) break;

		Point2f matchLoc = maxLoc;
		Rect2d roi = Rect2f(matchLoc, Size(patch.cols, patch.rows));

		double zero_base = 0.02;
		double roi_f = zero_base + roi_scaler;
		int x1 = roi.x - roi.width * roi_f > 0 ? roi.x - roi.width * roi_f : 0;
		int y1 = roi.y - roi.height* roi_f > 0 ? roi.y - roi.height* roi_f : 0;

		int x2 = roi.x + roi.width * roi_f > fus_map.cols ? fus_map.cols : roi.x + roi.width * roi_f;
		int y2 = roi.y + roi.height* roi_f > fus_map.rows ? fus_map.rows : roi.y + roi.height* roi_f;

		Rect2f roi_mask_fuse = Rect2f(Point2f(x1, y1), Point2f(x2, y2));
		fus_map(roi_mask_fuse) = 0;

		double roi_s = 1 + roi_f;
		int x3 = roi.x + roi_s * roi.width > _img.cols ? _img.cols : roi.x + roi_s * roi.width;
		int y3 = roi.y + roi_s * roi.height > _img.rows ? _img.rows : roi.y + roi_s * roi.height;

		Rect2f roi_mask = Rect2f(Point2f(x1 / scaler_f, y1 / scaler_f), Point2f(x3 / scaler_f, y3 / scaler_f));
		//rectangle(_img, roi_mask, CV_RGB(0, 0, 200));
		roi_mask_trip.push_back(roi_mask);
	}

	vector<Mat> sub_img;
	for (auto roi : roi_mask_trip)
	{
		sub_img.push_back(_img(roi));
	}
	for (size_t i = 0; i < sub_img.size(); ++i)
	{
		if (_patch.rows > sub_img[i].rows || _patch.cols > sub_img[i].cols)continue;
		matchTemplate(sub_img[i], _patch, fus_map, TM_CCORR_NORMED);
		normalize(fus_map, fus_map, 0, 1, NORM_MINMAX, -1);
		threshold(fus_map, fus_map, similarity, 255, THRESH_TOZERO);
		double maxVal; Point maxLoc;
		minMaxLoc(fus_map, NULL, &maxVal, NULL, &maxLoc);

		Point2f matchLoc = maxLoc;
		Rect2f roi = Rect2f(matchLoc, Size(_patch.cols, _patch.rows));

		roi.x += roi_mask_trip[i].x;
		roi.y += roi_mask_trip[i].y;
		roi_trip.push_back(roi);
	    //rectangle(_img, roi_trip[i], CV_RGB(0, 0, 128));
	}
	//src_trip = get_centre(_img, roi_trip);
	right_hand_based_tripple_sort(src_trip);
	return src_trip;
}

enum Pattern { CHESSBOARD = 1, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID , PATTERN_MAX};

void calcBoardCornerPositions(Size boardSize, Size2f squareSize, Mat & corners, Pattern patternType)
{
	Point3f base(125.63, 123.4 , 0);
	//Point3f base(0.00000000001, 0.00000000001,0);
	//Point3f base(0, 0, 0);
	switch (patternType)
	{
	case CHESSBOARD:
	case CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; ++i)
			for (int j = 0; j < boardSize.width; ++j)
				corners.push_back(Point3f(base.x + j * squareSize.width, base.y + i * squareSize.height , base.z));
		break;

	case ASYMMETRIC_CIRCLES_GRID:
		for (int i = 0; i < boardSize.height; i++)
			for (int j = 0; j < boardSize.width; j++)
				corners.push_back(Point3f((2 * j + i % 2)*squareSize.height, i*squareSize.width , base.z));
		break;
	default:
		break;
	}
}

Mat GetHomograph(Mat view, Size2f _real_size, Size chess_board_num)
{
	int chessBoardFlags = CALIB_CB_FAST_CHECK | CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
	Mat Oc_Points;
	Size boardSize = chess_board_num;//(7, 7);
	bool found = findChessboardCorners(view, boardSize, Oc_Points, chessBoardFlags);
	if (found)
	{
		cornerSubPix(view, Oc_Points, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
	}

	//Size2f _chess_board_real_size(50, 64.9);
	Size2f squareSize = _real_size; //(62.93, 65.1);
	Mat Ow_Points;
	calcBoardCornerPositions(boardSize, squareSize, Ow_Points, CHESSBOARD);

	Mat H = findHomography(Oc_Points, Ow_Points);
	H.convertTo(H, CV_32FC1);

#ifdef _DEBUG
	Mat Oc_view = view.clone();
	drawChessboardCorners(Oc_view, boardSize, Mat(Oc_Points), found);

	Mat Re_Ow_Points;
	vector<Point2f> _Oc_Points = Oc_Points;
	for (auto _X : _Oc_Points)
	{
		Mat X = (Mat_<float>(3, 1) << _X.x, _X.y, 1);
		Mat X_Ow = H * X;
		X_Ow = X_Ow / X_Ow.at<float>(Point(0, 2));
		Re_Ow_Points.push_back(Point2f(X_Ow.at<float>(Point(0, 0)), X_Ow.at<float>(Point(0, 1))));
    }
	Mat Oc2Ow_Review = view.clone();
	draw_circle(Oc2Ow_Review, Re_Ow_Points);
	
	vector<Point3f> _Ow_Points = Ow_Points;
	Mat Or_Ow_Points;
	for (auto p : _Ow_Points)
	{
		Or_Ow_Points.push_back( Point2f(p.x,p.y) );
	}
	Mat Ow_view = view.clone();
    draw_circle(Ow_view, Or_Ow_Points);

	Mat err = Or_Ow_Points - Re_Ow_Points;
	vector<Mat> x_y;
	split(err, x_y);
	Mat X = x_y[0];
	X = X.mul(X);
	Mat Y = x_y[1];
	Y = Y.mul(Y);
	Mat Z = X + Y;
	double  rms = sum(X + Y)[0] / X.rows;
	cout << "mean square error of H is  " << rms << "  in pixel units " <<endl;
#endif
	return H;
}

void calibrate_camera(Pattern type, Mat & view, Size board_num , Size2f _real_size, vector<Mat> & vec_mat)
{
	vec_mat.clear();
	Mat Oc_Points;
	bool found = false;
	switch (type)
	{
		case CHESSBOARD:
			found = findChessboardCorners(view, board_num, Oc_Points, CALIB_CB_FAST_CHECK | CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
			break;
		case CIRCLES_GRID:
		{
			SimpleBlobDetector::Params params;
			params.maxArea = 10000;
			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
			found = findCirclesGrid(view, board_num, Oc_Points, CALIB_CB_SYMMETRIC_GRID, detector);
			break;
		}
		case ASYMMETRIC_CIRCLES_GRID:
		{
			SimpleBlobDetector::Params params;
			params.maxArea = 10000;
			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
			found = findCirclesGrid(view, board_num, Oc_Points, CALIB_CB_ASYMMETRIC_GRID, detector);
			//drawChessboardCorners(view, board_num, Mat(Oc_Points), found);
			break;
		}
		default:
			found = false;
			break;
	}
	CHECK_EXP_RETURN(!(found && (Oc_Points.channels() == 2)));
	cornerSubPix(view, Oc_Points, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

	Mat Ow_Points;
	calcBoardCornerPositions(board_num, _real_size, Ow_Points, type);

	CHECK_EXP_RETURN(Ow_Points.empty() || Ow_Points.channels() != 3);

	vector<vector<Point3f> > Oc(1);
	vector<Mat> x_y;
	Mat z = Mat::zeros(Oc_Points.size(),CV_32FC1);
	split(Oc_Points, x_y);
	x_y.push_back(z);
	merge(x_y, z);
	Oc[0] = z;

	vector<vector<Point2f> > Ow(1);
	x_y.clear();
	split(Ow_Points, x_y);
	x_y.erase( --(x_y.end()));
	merge(x_y,z);
	Ow[0] = z;

	Size imageSize = view.size();
	Mat cameraMatrix, distCoeffs;
	vector<Mat> rvecs, tvecs;
	int flags = CALIB_FIX_PRINCIPAL_POINT;// | CALIB_ZERO_TANGENT_DIST | CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3;
	double rms = calibrateCamera(Oc, Ow, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);
	CHECK_EXP_RETURN(rms > 1);

	vec_mat.push_back(rvecs[0]);
	vec_mat.push_back(tvecs[0]);
	vec_mat.push_back(cameraMatrix);
	vec_mat.push_back(distCoeffs);

#ifdef _DEBUG
	cout << "calibrateCamera rms of  Levenberg-Marquardt is  " << rms << endl;
	Mat I, R, T, Rt;
	Mat _I = cameraMatrix.clone();
	_I.at<double>(Point(2, 2)) = 0;
	_I = _I.t();
	I = _I(Rect(0, 0, 3, 2));
	Mat Temp0 = (Mat_<double>(1,3) << 0, 0, 1);
	I.push_back(Temp0);
	Mat t = tvecs[0];
	double Zc = tvecs[0].at<double>(Point(0, 2));
	I = I / Zc;
	Temp0 = _I(Rect(0, 2, 3, 1));
	I.push_back(Temp0);
	I = I.t();

	Rodrigues(rvecs[0], R);
	hconcat(R, tvecs[0], Rt);
	Mat Temp1 = (Mat_<double>(1, 4) << 0, 0, 0, 1);
	Rt.push_back(Temp1);

	Mat _H = I * Rt;
	_H = _H.t();
	Mat H = _H(Rect(0, 0, 3, 2));
	Mat Temp2 = _H(Rect(0, 3, 3, 1));
	H.push_back(Temp2);
	H = H.t();

	vector<Point2f> Re_Ow_Points;
	{
		vector<Point2f> _Oc_Points = Oc_Points;
		for (auto p : _Oc_Points)
		{
			Mat XX = (Mat_<double>(3, 1) << p.x, p.y, 1);
			Mat XX_Oc = H * XX;
			Re_Ow_Points.push_back(Point2f(XX_Oc.at<double>(Point(0, 0)), XX_Oc.at<double>(Point(0, 1))));
		}
		vector<Point2f> _Ow_Points = Ow[0];
		Mat err = Mat(_Ow_Points) - Mat(Re_Ow_Points);
		vector<Mat> x_y;
		split(err, x_y);
		Mat X = x_y[0];
		X = X.mul(X);
		Mat Y = x_y[1];
		Y = Y.mul(Y);
		Mat Z = X + Y;
		double  rms2 = sum(X + Y)[0] / X.rows;
		cout << "mean square error of calibrateCamera and  IRt Oc to Ow is " << rms2 << "  in pixel units " << endl;
	}
	Mat Oc_view = view.clone();
	draw_circle(Oc_view, Oc_Points);
	//drawChessboardCorners(Oc_view, board_num, Mat(Oc_Points), found);

	Mat Ow_view = view.clone();
	draw_circle(Ow_view, Ow_Points);

	Mat Oc2Ow_Review = view.clone();
	Mat Re_Ow_Point = Mat(Re_Ow_Points);
	draw_circle(Oc2Ow_Review, Re_Ow_Point);

	Mat Re_Ow_Point2;
	perspectiveTransform(Oc_Points, Re_Ow_Point2, H);
	Mat Oc2Ow_Review2 = view.clone();
	draw_circle(Oc2Ow_Review2, Re_Ow_Point2);
#endif
	return;
}


void Oc_to_Ow(const vector<Point2f> & Oc_Points , const vector<Mat> & Homograph , vector<Point2f> & Ow_Points)
{
	CHECK_EXP_RETURN(Oc_Points.empty());
	Mat Oc3 = Mat(Oc_Points);
	CHECK_EXP_RETURN(Oc3.channels() != 2);
	CHECK_EXP_RETURN(Mat(Ow_Points).channels() != 2);

	vector<Point3f> Oc;
	vector<Mat> x_y;
	Mat z = Mat::zeros(Oc3.size(), CV_32FC1);
	split(Oc3, x_y);
	x_y.push_back(z);
	merge(x_y, z);
	Oc = z;

	CHECK_EXP_RETURN(Mat(Oc).channels() != 3);
	CHECK_EXP_RETURN(Homograph.size() != 4);
	projectPoints(Oc, Homograph[0], Homograph[1], Homograph[2], Homograph[3], Ow_Points);
}

Point2f re_mapping(Point2f _X, Mat A, Mat H)
{
	//convertPointsToHomogeneous
	Mat X = (Mat_<float>(3, 1) << _X.x, _X.y, 1);
	Mat K = (A * X);
	Mat X_Ow = H * (A * X);
	//convertPointsFromHomogeneous(X_Ow, X_Ow);
	X_Ow = X_Ow / X_Ow.at<float>(Point(0, 2));
	return Point2f(X_Ow.at<float>(Point(0, 0)), X_Ow.at<float>(Point(0, 1)));
}

vector<Point2f> re_mapping2(vector<Point2f> X, Mat A, Mat H)
{
	vector<Point2f> Y;
	for (auto x : X)
	{
		Mat X = (Mat_<float>(3, 1) << x.x, x.y, 1);
		Mat K = (A * X);
		Mat X_Ow = H * (A * X);
		//convertPointsFromHomogeneous(X_Ow, X_Ow);
		X_Ow = X_Ow / X_Ow.at<float>(Point(0, 2));
		Y.push_back(Point2f(X_Ow.at<float>(Point(0, 0)), X_Ow.at<float>(Point(0, 1))));
	}
	return Y;
}


vector<Point2f> Obj_to_Ow(const vector<Point2f> & Obj_Points,vector<Point2f> & src_tri ,const vector<Mat> & vec_mat)
{
	vector<Point2f> Ow_Points;
	Mat A;
	A = get_normed_affine_transform(src_tri); //图像坐标的 主方向 仿射变换阵
	CHECK_EXP_RETURN(A.empty() , Ow_Points);

	vector<Point2f> Oc_Points;
	Oc_Points = re_mapping(Obj_Points, A); //物体坐标 到 图像坐标的 仿射变换;
	CHECK_EXP_RETURN(Oc_Points.empty(), Ow_Points);

	//Oc_to_Ow(Oc_Points, vec_mat, Ow_Points); //相机图像坐标 到 世界坐标的 透视变换
	Ow_Points = Oc_Points;
	return Ow_Points;
}

void calibrate_camera2(Mat view, Size2f _real_size, Size chess_board_num , vector<Mat> & vec_mat)
{
	int chessBoardFlags = CALIB_CB_FAST_CHECK | CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
	Mat Oc_Points;
	Size boardSize = chess_board_num;
	bool found = findChessboardCorners(view, boardSize, Oc_Points, chessBoardFlags);
	if (found)
	{
		cv::cornerSubPix(view, Oc_Points, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
	}

	Size2f squareSize = _real_size;
	Mat Ow_Points;
	calcBoardCornerPositions(boardSize, squareSize, Ow_Points, CHESSBOARD);

	vector<vector<Point3f> > objectPoints(1);
	objectPoints[0] = Ow_Points;
	Size imageSize = view.size();
	vector<vector<Point2f> > imagePoints(1);
	imagePoints[0] = Oc_Points;
	Mat cameraMatrix, distCoeffs;
	vector<Mat> rvecs, tvecs;

	int flags = CALIB_FIX_PRINCIPAL_POINT;// | CALIB_ZERO_TANGENT_DIST | CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3;
	double rms = calibrateCamera(objectPoints,imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs , flags);

	if (rms > 1) return;

	vec_mat.push_back(rvecs[0]);
	vec_mat.push_back(tvecs[0]);
	vec_mat.push_back(cameraMatrix);
	vec_mat.push_back(distCoeffs);

#ifdef _DEBUG
	vector<Point2f> Re_Oc_Points;
	projectPoints(objectPoints[0], rvecs[0], tvecs[0] , cameraMatrix , distCoeffs , Re_Oc_Points);

	cout << "rms of  Levenberg-Marquardt is  " << rms << endl;
	Mat err = Mat(Oc_Points) - Mat(Re_Oc_Points);
	vector<Mat> _x_y;
	split(err, _x_y);
	Mat X = _x_y[0];
	X = X.mul(X);
	Mat Y = _x_y[1];
	Y = Y.mul(Y);
	Mat Z = X + Y;
	double  rms2 = sum(X + Y)[0] / X.rows;
	cout << "mean square error of calibrateCamera and  projectPoints Ow to Oc is  " << rms2 << "  in pixel units " << endl;

	Mat I, R, T, Rt;
	Mat _I = cameraMatrix.clone();
	_I.at<double>(Point(2, 2)) = 0;
	_I = _I.t();
	I = _I(Rect(0, 0, 3, 2));
	Mat Temp0 = (Mat_<double>(1, 3) << 0, 0, 1);
	I.push_back(Temp0);
	Mat t = tvecs[0];
	double Zc = tvecs[0].at<double>(Point(0, 2));
	I = I / Zc;
	Temp0 = _I(Rect(0, 2, 3, 1));
	I.push_back(Temp0);
	I = I.t();

	Rodrigues(rvecs[0], R);
	hconcat(R, tvecs[0], Rt);
	Mat Temp1 = (Mat_<double>(1, 4) << 0, 0, 0, 1);
	Rt.push_back(Temp1);

	Mat _H = I * Rt;
	_H = _H.t();
	Mat H = _H(Rect(0, 0, 3, 2));
	Mat Temp2 = _H(Rect(0, 3, 3, 1));
	H.push_back(Temp2);
	H = H.t();
	H = H.inv();


	vector<Point3f> Re_Ow_Points;
	{
		for (auto p : imagePoints[0])
		{
			Mat XX = (Mat_<double>(3, 1) << p.x, p.y, 1);
			Mat XX_Ow = H * XX;
			Re_Ow_Points.push_back(Point3f(XX_Ow.at<double>(Point(0, 0)), XX_Ow.at<double>(Point(0, 1)) , 0));
		}
		Mat err = Mat(Ow_Points) - Mat(Re_Ow_Points);
		vector<Mat> x_y;
		split(err, x_y);
		Mat X = x_y[0];
		X = X.mul(X);
		Mat Y = x_y[1];
		Y = Y.mul(Y);
		Mat Z = X + Y;
		double  rms2 = sum(X + Y)[0] / X.rows;
		cout << "mean square error of calibrateCamera and  IRt Oc to Ow is  " << rms2 << "  in pixel units " << endl;
	}
	Mat Oc_view = view.clone();
	drawChessboardCorners(Oc_view, boardSize, Mat(Oc_Points), found);

	Mat Ow_view = view.clone();
	draw_circle(Ow_view, Ow_Points);

	Mat Oc2Ow_Review = view.clone();
	Mat Re_Ow_Point = Mat(Re_Ow_Points);
	draw_circle(Oc2Ow_Review, Re_Ow_Point);

	Mat Re_Ow_Point2;
	perspectiveTransform(Oc_Points, Re_Ow_Point2, H);
	Mat Oc2Ow_Review2 = view.clone();
	draw_circle(Oc2Ow_Review2, Re_Ow_Point2);
#endif
	return;
}

enum err_type { SUCCESS = 0 ,PATTERN_VIEW_INVALID = 1 , IMAGE_INVALID = 2 ,IMAGE_TEMP_INVALID = 3, CALIBRATE_FAILED = 4 ,
	            MATCHING_TRIPPLE_FAILED = 5 , OBI_EMPTY = 6 , REMMPING_FAILED = 7 , PATTERN_INVALID , CALIBRATE_SUCCESS , SYS_CONFIG_ERROR , SYS_CALIBRATE_CONFIG_ERROR
};

class img_config
{
public:
	img_config(Mat _img = Mat() , Mat _temp  = Mat(), vector<Point2f> _obj_points = vector<Point2f>() ,vector<Mat> _vec_mat_H = vector<Mat>())
	{
		if (!_img.empty())
		{
			if (_img.channels() != 1) cvtColor(_img, _img, CV_BGR2GRAY);;
			img = _img;
		}
		else 
		{ 
			return; 
		}

		/*if (!_temp.empty())
		{
			if (_temp.channels() != 1) cvtColor(_temp, _temp, CV_BGR2GRAY);;
			img_config::temp = _temp;
		}
		if (!_vec_mat_H.empty())
		{
			vec_mat_H = _vec_mat_H;
		}

		if ((_scaler >= 0.01 && _scaler <= 100))
		{
			scaler = _scaler;
			cv::resize(img_config::temp, img_config::temp, Size(), scaler, scaler, INTER_CUBIC);
			cv::resize(img, img, Size(), scaler, scaler, INTER_CUBIC);
		}*/

		if (!_obj_points.empty() && !_temp.empty() && !_vec_mat_H.empty())
		{
			vector<Point2f> src_tri;
			src_tri = matching_template_accelerated_knn(img, img_config::temp);
			if (src_tri.size() == 3)
			{
				Mat A = get_normed_affine_transform(src_tri, false);
				if (!A.empty())
				{
					//draw_circle(_img, src_tri, Scalar(66));
					img_config::obj_points = re_mapping(_obj_points, A);
					//draw_circle(_img, img_config::obj_points, Scalar(199));
					//Mat A = get_normed_affine_transform(src_tri);
					//vector<Point2f> obj_points2 = re_mapping(img_config::obj_points, A);
				}
				else
				{
					img_config::obj_points.clear();
				}
			}
			else
			{
				img_config::obj_points.clear();
			}
		}
	};

	~img_config() {};

	bool is_config_valid()
	{
		return !(img.empty() || img_config::temp.empty() || img_config::obj_points.empty());
	}

	Mat img;
	static Mat temp;
	static vector<Point2f> obj_points;
	static vector<Mat> vec_mat_H;
	vector<Point2f> ow_points;
};

Mat img_config::temp = Mat();
vector<Point2f> img_config::obj_points = vector<Point2f>();
vector<Mat> img_config::vec_mat_H = vector<Mat>();

class view_config
{
public:
	view_config(Mat _view = Mat(), Pattern _type = PATTERN_MAX,  Size2f _one_real_size = Size2f() , Size _pattern_size = Size())
	{
		if ( !(_view.empty() || _one_real_size.empty() || _pattern_size.empty() || _type >= PATTERN_MAX))
		{
			if (_view.channels() != 1) cvtColor(_view, _view, CV_BGR2GRAY);
			view = _view;
			type = _type;
			one_real_size = _one_real_size;
			pattern_size = _pattern_size;
		}
	};

	bool is_config_valid()
	{
		return !(view.empty() || one_real_size.empty() || pattern_size.empty() || type >= PATTERN_MAX);
	}

	~view_config() {};

	Mat view;
	Pattern type;
	Size pattern_size;
	Size2f one_real_size;
	vector<Mat> vec_mat;
};

err_type Laser_Printer_System(img_config & sys_config , view_config sys_calibrate = view_config())
{
	static vector<Mat> vec_mat; //透视变换阵
	if (sys_calibrate.is_config_valid())
	{
		Mat view = sys_calibrate.view;
		Pattern type = sys_calibrate.type;
		Size pattern_size = sys_calibrate.pattern_size;
		Size2f one_real_size = sys_calibrate.one_real_size;

		vec_mat.clear();
		for (int i = 0, try_times = 3; i < try_times; ++i)
		{
			calibrate_camera(type, view, pattern_size,one_real_size, vec_mat); //相机 矫正 标定
			CHECK_EXP_RETURN(vec_mat.size() == 4, CALIBRATE_SUCCESS);
		}
	}
	CHECK_EXP_RETURN(vec_mat.size() != 4, SYS_CALIBRATE_CONFIG_ERROR);

	CHECK_EXP_RETURN( !sys_config.is_config_valid() , SYS_CONFIG_ERROR);
	Mat img = sys_config.img;
	Mat temp = sys_config.temp;
	vector<Point2f> obj_points = sys_config.obj_points;

	vector<Point2f> src_tri;
	src_tri = matching_template_accelerated_knn(img, temp);
	CHECK_EXP_RETURN(src_tri.size() != 3, MATCHING_TRIPPLE_FAILED);

#ifdef _DEBUG
	
	//draw_circle(_img, src_tri);
	//set_lable(img, to_string((float)time) + string(" ms "), src_tri[0]);
	//imshow("matching _img", _img);
#endif

	vector<Point2f> ow_points = Obj_to_Ow(obj_points, src_tri, vec_mat);
	CHECK_EXP_RETURN(ow_points.empty(), REMMPING_FAILED);
	sys_config.ow_points = ow_points;
//#ifdef _DEBUG
	Mat _img = img.clone();
	draw_circle(_img, sys_config.ow_points , Scalar(0));
	cv::resize(_img, _img, Size(), 0.5, 0.5, INTER_CUBIC);
	imshow("matching _img", _img);
//#endif
	return SUCCESS;
}


Mat img_cycle_trans(Mat & srcImage, int step = 1)
{
	auto type = srcImage.type();
	srcImage.convertTo(srcImage , CV_32FC1);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	Mat res(srcImage.size(), srcImage.type() , Scalar(0));

	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			int x = (j + step) % nCols;
			int y = i ;
			res.at<float>(y, x) = srcImage.at<float>(i,j);
		}
	}
	res.convertTo(res, type);
	srcImage.convertTo(srcImage, type);
	return res;
}

VideoCapture gen_video_from_img(string img_path, int frame_num = 1200,int frame_rate = 30, double speed = 0.01 , double black_edge_expend = 1.2)
{
	Mat img = imread(img_path, IMREAD_GRAYSCALE);
	img = imrotate(img , 8);
	Mat expend_img = Mat::zeros(Size(img.cols * black_edge_expend, img.rows), img.type());
	Rect2f roi = Rect(Point2f(expend_img.cols / 2.0, expend_img.rows / 2.0) - Point2f(img.cols / 2.0, img.rows / 2.0), img.size());
	img.copyTo(expend_img(roi));
	img = expend_img;
	img = 255 - img;
	VideoWriter VW;
	VideoCapture VC;
	string video_path = string(img_path.begin(), img_path.end() - 3) + "avi";

	if (VW.open(video_path, CV_FOURCC('X', 'V', 'I', 'D'), frame_rate, img.size(), false))
	{
		Mat f_pre = img;
		for (; frame_num--;)
		{
			VW << f_pre;
			f_pre = img_cycle_trans(f_pre, speed * img.cols);

			Mat t = f_pre.clone();
			resize(t, t, Size(), 0.25, 0.25, INTER_AREA);
			imshow("frames", t);
			if ((waitKey(1) & 0xff) == 27) break;
		}
		VW.release();
		VC.open(video_path);
	}
    return VC;
}


vector<Mat>& gen_video(Mat img, int frame_num = 300, double speed = 0.001, int frame_rate = 30, string vedio_name = "demo")
{
	static vector<Mat> frames;
	frames.clear();
	string video_path = "../../data/" + vedio_name + string(".avi");
	ifstream fin(video_path);
	if (!fin)
	{

		Mat bg = img.clone();
	
		bg = bg.t();
		bg.push_back(bg);
		bg.push_back(bg);
		bg = bg.t() * 0.5;
		bg = 0;
		Mat f_pre = img.clone();
		resize(bg, bg, f_pre.size());

		VideoWriter V;
		V.open(video_path, CV_FOURCC('X', 'V', 'I', 'D'), frame_rate, f_pre.size(), 0);
		if (!V.isOpened()) cout << "Could not open the output video for write " << endl;
		for (; frame_num--;)
		{
			Mat f_next = img_cycle_trans(f_pre, speed * img.cols);
			f_pre = f_next;
			Mat fs = f_pre.clone();
			Mat bgs = bg.clone();
			fs.convertTo(fs, CV_8UC1);
			fs.copyTo(bgs, Mat(fs > 127));
			
			bgs.convertTo(bgs, CV_8UC1);
			V << bgs;

			//frames.push_back(bgs);

#ifdef _DEBUG
			Mat t = bgs.clone();
			resize(t, t, Size(), 0.25, 0.25, INTER_AREA);
			imshow("frames", t);
			if ((waitKey(1) & 0xff) == 27) break;
#endif
		}
		V.release();
	}
	/*else
	{
		VideoCapture VC;
		VC.open(video_path);
		if (!VC.isOpened())
		{
			cout << "Read video Failed !" << endl;
			return frames;
		}
		Mat frame;
		int frame_num = 0;
		while (true)
		{
			VC.read(frame);
			if (frame.empty())break;
			cvtColor(frame, frame, CV_BGR2GRAY);
			frames.push_back(frame);
		} 
	}*/
	return frames;
}


//matching_template_accelerated(objs, diff_set);

class ml_obj
{
public:

	double area_th;
	static Obj_Set std;
	static double lens_per_pixel;
	static map<size_t, Mat> objs;
	static map<size_t, diff_map> diff_set;

	static Mat mark;
	static Mat A;
	static double obj_area;
	static map<size_t, Point2f> Oxy;
	static map<size_t, Point2d> obj_centre;
	static map<size_t, Mat> objs_mark;
	static vector<Point2f> obj_points;
	static vector<Mat> vec_mat_H;
	static map<size_t, vector<Point2f>> Ow_objs;

	Mat & frame;
	double & tracking_scaler;
	map<size_t, Rect2d> & tracking_box;

	ml_obj( Mat & _frame ,  double & _tracking_scaler , map<size_t, Rect2d> & _tracking_box) : frame(_frame) , tracking_scaler(_tracking_scaler), tracking_box(_tracking_box){};
	
	double CalculatWidthDiff_Thin(Mat obj_img, Mat std_temp, multimap<double, Rect2f, greater<double>>& indesity) {
		double max_width = 0;
		Mat sub_img, thin, sub_edge_img, diff;
		absdiff(std_temp, obj_img, diff);
		for (auto& v : indesity)
		{
			sub_img = diff(v.second).clone();
			Mat sub_s(Size(sub_img.size().width * 1.2, sub_img.size().height * 1.2), sub_img.type(), Scalar(0));
			Point2f tl = Point2f((sub_s.cols - sub_img.cols) / 2.0, (sub_s.rows - sub_img.rows) / 2.0);
			Rect2f roi = Rect(tl, sub_img.size());
			sub_img.copyTo(sub_s(roi));
			Mat thin = thin_3(sub_s);
			double len = sum(thin)[0];
			double area = sum(sub_s)[0] / 255;
			if (len < 1e-3)max_width = min(sub_img.rows, sub_img.cols);
			else max_width = area / len;
			if (max_width >= min(sub_img.cols, sub_img.rows))
			{
				sub_s = sub_s.t();
				thin = skeleton(sub_s);
				len = sum(thin)[0];
				if (len < 1e-3)max_width = min(sub_img.cols, sub_img.rows);
				else max_width = area / len;
			}
			if (max_width >= min(sub_img.cols, sub_img.rows)) max_width = min(sub_img.cols, sub_img.rows);
			//max_width /= scaler_f2;
			break;

		}
		return max_width;
	}

	double CalculateWidthDiff_Gradient(Mat obj_img, Mat std_temp, multimap<double, Rect2f, greater<double>>& indesity)
	{
		double max_width = 0;
		Mat contours_objection_and_template = Mat::zeros(std_temp.size(), std_temp.type());
		vector<vector<Point>> contours_std_temp;
		Mat std_temp_clone = std_temp.clone();
		std_temp_clone.convertTo(std_temp_clone, CV_8UC1);
		threshold(std_temp_clone, std_temp_clone, 20, 255, THRESH_TOZERO);
		cv::findContours(std_temp_clone, contours_std_temp, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		Mat obj_img_clone = obj_img.clone();
		obj_img_clone.convertTo(obj_img_clone, CV_8UC1);
		threshold(obj_img_clone, obj_img_clone, 20, 255, THRESH_TOZERO);
		vector<vector<Point>> contours_obj_img;
		cv::findContours(obj_img_clone, contours_obj_img, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		cv::drawContours(contours_objection_and_template, contours_std_temp, -1, 255, 1, 8);
		cv::drawContours(contours_objection_and_template, contours_obj_img, -1, 255, 1, 8);
		/*Mat_<double> std_temp_show = diff.clone();
		std_temp_show = std_temp_show - contours_objection_and_template;*/
		Mat diff_img;
		absdiff(std_temp, obj_img, diff_img);
		for (auto& v : indesity)
		{

			Mat sub_img = diff_img(v.second).clone();
			Mat sub_edge_img = contours_objection_and_template(v.second).clone();
			if (sub_img.type() != CV_8UC1)sub_img.convertTo(sub_img, CV_8UC1);
			if (sub_edge_img.type() != CV_8UC1)sub_edge_img.convertTo(sub_edge_img, CV_8UC1);
			Mat_<double> sobel_x = Mat::zeros(sub_img.size(), CV_32SC1);
			Mat_<double> sobel_y = Mat::zeros(sub_img.size(), CV_32SC1);
			Mat_<double> theta = Mat::zeros(sub_img.size(), CV_32SC1);
			max_width = 0;
			//double x_gradient, y_gradient,oritentation_gradient;
			int count_num = 0;
			for (size_t row = 1; row < sub_edge_img.rows - 1; row++) {
				for (size_t col = 1; col < sub_edge_img.cols - 1; col++) {

					if (row<1 || col<1 || row>sub_img.rows - 2 || col>sub_img.cols - 2) continue;

					if (sub_edge_img.at<uchar>(row, col) < 1e-3)continue;
					else
					{
						uchar* up_p = sub_img.ptr<uchar>(row - 1);
						uchar* p = sub_img.ptr<uchar>(row);
						uchar* down_p = sub_img.ptr<uchar>(row + 1);
						//compute the x,y gradient
						sobel_x(row, col) = (double)(up_p[col + 1] + p[col + 1] * 2 + down_p[col + 1] - up_p[col - 1] - p[col - 1] * 2 - down_p[col - 1]);
						sobel_y(row, col) = (double)(down_p[col - 1] + down_p[col] * 2 + down_p[col + 1] - up_p[col - 1] - up_p[col] * 2 - up_p[col + 1]);
						theta(row, col) = atan2(sobel_y(row, col), sobel_x(row, col));//gradient oritentation(-pi~pi)

						int max_iter = 0;
						size_t deta_x = 1;
						size_t deta_y = 1;
						double x = 0;
						double y = 0;
						int judge_orientation = 0;
						if (CV_PI / 4 > theta(row, col) && theta(row, col) > -CV_PI / 4) {
							max_iter = sub_img.cols - col - 1;
							judge_orientation = 1;
						}
						else if (CV_PI * 3 / 4 > theta(row, col) && theta(row, col) >= CV_PI / 4) {
							max_iter = sub_img.rows - row - 1;
							judge_orientation = 2;
						}
						else if (-CV_PI * 3 / 4 < theta(row, col) && theta(row, col) <= -CV_PI / 4) {

							max_iter = row + 1;
							judge_orientation = 3;
						}
						else {
							max_iter = col + 1;
							judge_orientation = 4;
						}
						double cur_width = sub_img.at<uchar>(row, col);
						for (; max_iter > 0; max_iter--)
						{
							if (1 == judge_orientation) {
								x = col + deta_x;
								y = tan(theta(row, col)) * (x - col) + row;
								deta_x++;
							}
							else if (2 == judge_orientation) {
								y = row + deta_y;
								x = tan(CV_PI / 2 - theta(row, col)) * (y - row) + col;
								deta_y++;
							}
							else if (3 == judge_orientation) {
								y = row - deta_y;
								x = tan(CV_PI / 2 - theta(row, col)) * (y - row) + col;
								deta_y++;
							}
							else if (4 == judge_orientation)
							{
								x = col - deta_x;
								y = tan(theta(row, col)) * (x - col) + row;
								deta_x++;
							}
							else break;

							if (x > sub_img.cols - 2 || y > sub_img.rows - 2 || x < 1 || y < 1)break;
							if (sub_img.at<uchar>(cvRound(y), cvRound(x)) > 0) {
								double weight1 = 0.5;
								double weight2 = 0.5;
								if (deta_x > 1) {
									weight1 = 1 - y + cvFloor(y);
									weight2 = 1 - weight1;
									cur_width += weight1 * sub_img.at<uchar>(cvFloor(y), (int)x) + weight2 * sub_img.at<uchar>(cvFloor(y) + 1, x);
								}
								if (deta_y > 1) {
									weight1 = 1 - x + cvFloor(x);
									weight2 = 1 - weight1;
									cur_width += weight1 * sub_img.at<uchar>((int)y, cvFloor(x)) + weight2 * sub_img.at<uchar>((int)y, cvFloor(x) + 1);
								}

							}
							else
							{
								cur_width /= 255;
								if (deta_x > 1)cur_width /= abs(cos(theta(row, col)));
								if (deta_y > 1)cur_width /= abs(sin(theta(row, col)));
								double cur_pre_width_ratio = 0;
								double judgement_ratio = 1.5;
								if (max_width != 0 && count_num != 0)
									cur_pre_width_ratio = cur_width / (max_width / count_num);
								if (cur_pre_width_ratio > judgement_ratio)break;
								else if (0 < cur_pre_width_ratio && cur_pre_width_ratio < 1 / judgement_ratio) {
									max_width = cur_width;
									count_num = 1;
								}
								else
								{
									max_width += cur_width;
									count_num++;
								}	
							}
							break;
						}
					}
				}
			}
			if (count_num == 0)max_width = min(sub_img.rows, sub_img.cols);
			else max_width /= count_num;
			if (max_width >= min(sub_img.cols, sub_img.rows)) max_width = min(sub_img.cols, sub_img.rows);
			break;
		}
		return max_width;
	}

	double CalculateWidthDiff_Edge(Mat obj_img, Mat std_temp, multimap<double, Rect2f, greater<double>>& indesity) {
		double max_width = 0.0500;
		Mat contours_objection_and_template = Mat::zeros(std_temp.size(), std_temp.type());
		vector<vector<Point>> contours_std_temp;
		Mat std_temp_clone = std_temp.clone();
		std_temp_clone.convertTo(std_temp_clone, CV_8UC1);
		threshold(std_temp_clone, std_temp_clone, 20, 255, THRESH_TOZERO);
		cv::findContours(std_temp_clone, contours_std_temp, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		Mat obj_img_clone = obj_img.clone();
		obj_img_clone.convertTo(obj_img_clone, CV_8UC1);
		threshold(obj_img_clone, obj_img_clone, 20, 255, THRESH_TOZERO);
		vector<vector<Point>> contours_obj_img;
		cv::findContours(obj_img_clone, contours_obj_img, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		cv::drawContours(contours_objection_and_template, contours_std_temp, -1, 255, 1, 8);
		cv::drawContours(contours_objection_and_template, contours_obj_img, -1, 255, 1, 8);
		/*Mat_<double> std_temp_show = diff.clone();
		std_temp_show = std_temp_show - contours_objection_and_template;*/
		Mat diff_img;
		absdiff(std_temp, obj_img, diff_img);
		for (auto& v : indesity)
		{
			Rect rect = v.second;
			Mat sub_img = diff_img(rect).clone();

			double area = sum(sub_img)[0] / 255;
			if (area > 0.9 * sub_img.size().area()) {
				max_width = max(sub_img.rows, sub_img.cols);
				return max_width;
			}
			double radius = (double)sqrtf(rect.height * rect.height + rect.width * rect.width) / 2.0;
			Point2f circle_center(rect.x + (float)(rect.width / 2.0), rect.y + (float)(rect.height / 2.0));
			Rect2f circle_rect(Point2f(circle_center.x - radius, circle_center.y - radius), Size2f(2 * radius, 2 * radius));
			/*circle_rect.x = circle_center.x - radius;
			circle_rect.y = circle_center.y - radius;
			circle_rect.width = circle_rect.height = 2 * radius;*/
			if (circle_rect.x<0 || circle_rect.y<0 || (circle_rect.x + circle_rect.width)>diff_img.cols - 1 ||
				(circle_rect.y + circle_rect.height)>diff_img.rows - 1) {
				cout << "Rectangular box crossing, please reset the rectangular box size" << endl;
				return max_width;
			}
			sub_img = diff_img(circle_rect).clone();
			Mat sub_contours_img = contours_objection_and_template(circle_rect).clone();
			if (sub_img.type() != CV_8UC1)sub_img.convertTo(sub_img, CV_8UC1);
			if (sub_contours_img.type() != CV_8UC1)sub_contours_img.convertTo(sub_contours_img, CV_8UC1);
			for (int row = 0; row < sub_img.rows; row++) {
				for (int col = 0; col < sub_img.cols; col++) {
					if (sqrtf((col - sub_img.cols / 2.0) * (col - sub_img.cols / 2.0) + (row - sub_img.rows / 2.0) * (row - sub_img.rows / 2.0)) < radius)continue;
					else
					{
						sub_img.at<uchar>(row, col) = 0;
						sub_contours_img.at<uchar>(row, col) = 0;
					}
				}
			}
			//rectangle(diff_img, circle_rect, Scalar(0), 1, LINE_4);
			area = sum(sub_img)[0] / 255;
			double length = sum(sub_contours_img)[0] / (2 * 255);
			if (length<1e-3 && area>CV_PI * radius * radius * 0.9) {
				max_width = 2 * radius;
				return max_width;
			}
			if (length < 1e-3 && area < 1) {
				max_width = 0;
				return max_width;
			}
			if (area < CV_PI * radius * radius / 3)max_width = area / length;
			else if (CV_PI * radius * radius / 3 <= area && area < CV_PI * radius * radius * 2 / 3) {
				if (length < 2 * radius) {
					length = (length + 2 * radius) / 2.0;
				}
				max_width = area / length;
			}
			else
			{
				length = 2 * radius;
				max_width = area / length;
			}
			if (max_width >= min(sub_img.cols, sub_img.rows)) max_width = min(sub_img.cols, sub_img.rows);
			break;
		}
		return max_width;
	}


	double CalculateWidthDiff(Mat obj_img, Mat std_temp, multimap<double, Rect2f, greater<double>>& indesity)
	{
#define	chooses  2 //skeleton;gradient;contours
		//chooses = 2;
		//static int chooses = 1;
		double max_width = 0;
#if (chooses == 0)
		max_width = CalculatWidthDiff_Thin(obj_img, std_temp, indesity);
#elif (chooses == 1)
		max_width = CalculateWidthDiff_Gradient(obj_img, std_temp, indesity);
#elif (chooses == 2)
		max_width = CalculateWidthDiff_Edge(obj_img, std_temp, indesity);
#endif
		return max_width;
	}

void locate_tripple_mark(const size_t & n)
	{
		Point2f _Oxy = Oxy[n];
		Mat & _obj = objs_mark[n];
		Mat & _mark = mark;
		if (_obj.empty() || _mark.empty() || vec_mat_H.empty())
		{
#ifdef _DEBUG
			cout << "_obj . _mark . vec_mat_H . is empty ! " << endl;
#endif
			return;
		}
		vector<Point2f> src_tri;	
		src_tri = matching_template_accelerated_knn(_obj, _mark);
		if (src_tri.size() == 3)
		{
			vector<Point2f> ow_points = Obj_to_Ow(obj_points, src_tri, vec_mat_H);
			if (ow_points.empty())
			{
#ifdef _DEBUG
				cout << " Obj_to_Ow transfer failed ! " << endl;
#endif
				return;
			}
			else
			{
				mark_mxt.lock();
				Ow_objs[n] = ow_points;
				objs_mark.erase(n);
				Oxy.erase(n);
				mark_mxt.unlock();
				return;
			}
		}
		else
		{
#ifdef _DEBUG
			cout << "  matching tripple mark failed ! " << endl;
#endif
			return;
		}
	}

Mat match_features(Mat img_1, Mat img_2)
{
	img_1.convertTo(img_1, CV_8UC1);
	img_2.convertTo(img_2, CV_8UC1);
	//Ptr<Feature2D> orb = xfeatures2d::SIFT::create();
	Ptr<ORB> orb = ORB::create(3000);
	//orb->setFastThreshold(0);

	vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;

	orb->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);
	orb->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

	//Mat ShowKeypoints1, ShowKeypoints2;
	//drawKeypoints(img_1, keypoints_1, ShowKeypoints1);
	//drawKeypoints(img_2, keypoints_2, ShowKeypoints2);
	//imshow("Result_1", ShowKeypoints1);
	//imshow("Result_2", ShowKeypoints2);

	vector<DMatch> matchesAll, matchesGMS;
	BFMatcher matcher(NORM_HAMMING, true);

	matcher.match(descriptors_1, descriptors_2, matchesAll);
	cout << "matchesAll: " << matchesAll.size() << endl;
	xfeatures2d::matchGMS(img_1.size(), img_2.size(), keypoints_1, keypoints_2, matchesAll, matchesGMS , true ,false , 4);
	cout << "matchesGMS: " << matchesGMS.size() << endl;

	Mat finalMatches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matchesGMS, finalMatches, Scalar::all(-1), Scalar::all(-1), vector<char>());
	imshow("Matches GMS", finalMatches);
	imwrite("MatchesGMS.jpg", finalMatches);
	waitKey(0);
	return Mat();
}

bool is_in_laser_region(Rect2f & obj_roi_frame)
{
	Rect2f laser_region( Point2f(frame.cols / 8.0 * 4.0 ,frame.rows / 4.0 * 0) ,Point2f( frame.cols / 8.0 * 8.0 , frame.rows / 4.0 * 4.0));
	Mat x = frame;
	//rectangle(frame, laser_region, Scalar(255), 1, LINE_4);
	//rectangle(frame, obj_roi_frame, Scalar(255), 1, LINE_4);
	return (obj_roi_frame & laser_region).area()  >= obj_roi_frame.area();
}

	void matching_template_accelerated(const size_t & n)
	{	
		map<size_t, Obj_Set > obj_st;
		obj_st[n] =  Obj_Set( n,  objs[n].clone() );

		map<size_t, double> obj_master_dir;
		map<size_t,Point2f> obj_center;
		map<size_t, double> obj_scaler;
			
		for (auto & _obj : obj_st)
		{
			Obj_Set & obj = _obj.second;
			Mat & obj_img = obj.obj_img;
			
			vector<vector<Point>> contours;
			cv::findContours(obj_img, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			
			double area_total = 0;
			double _obj_area = 0;
			for (auto& c : contours)
			{
				area_total += moments(c, true).m00;
			}

			Point2f master_vec = Point2f(0, 0);
			vector<Point2f> avg_cen;
			for (auto & c : contours)
			{
				Moments Mpq = moments(c, true);
				double area_rate = Mpq.m00 / area_total;
				if (area_rate < area_th)
				{
					obj_img(boundingRect(c)) = 0;
					continue;
				}
				else
				{
					_obj_area += Mpq.m00;
				}
				Point2f centre = Point2f(static_cast<float>(Mpq.m10 / Mpq.m00), static_cast<float>(Mpq.m01 / Mpq.m00));
				avg_cen.push_back(centre * area_rate);
				master_vec += getOrientation(c, obj_img, centre);
				obj.std_obj.insert(obj.std_obj.end(), c.begin(), c.end());
			}
			Point2f cen = Point2f(0, 0);
			for (size_t i = 0; i < avg_cen.size(); ++i)
			{
				cen += avg_cen[i];
			}
			obj.centre = cen;
			double master_dir = atan2(master_vec.y, master_vec.x) * 180 / CV_PI;
			obj.master_dir = master_dir;
			obj.master_vec = master_vec;

			size_t _n = _obj.first;
			obj_center[_n] = cen;
			obj_scaler[_n] = sqrt(_obj_area / ml_obj::obj_area);
		}

		Point2f center_max = Point2f(0, 0);
		Size max_size(0 ,0);

		for (auto & os : obj_st)
		{
			size_t _n = os.first;
			Obj_Set & obj = os.second;
			obj.ang_diff = obj.master_dir - std.master_dir;
			obj_master_dir[_n] = obj.ang_diff;
			Mat_<double> R = getRotationMatrix2D(obj.centre, obj.ang_diff, obj_scaler[_n]);

			vector<Point2f> std_obj = obj.std_obj;
			Rect2f rect = boundingRect(std_obj);
			Point2f vertices[4];
			vertices[0] = rect.tl(); vertices[1] = Point2f(rect.br().x, rect.tl().y); vertices[2] = rect.br(); vertices[3] = Point2f(rect.tl().x, rect.br().y);

			double neg4_x = 0, neg4_y = 0;
			double max4_x = 0, max4_y = 0;
			for (int i = 0; i < 4; i++)
			{
				Mat X = (Mat_<double>(3, 1) << vertices[i].x, vertices[i].y, 1);
				X = (R * X);
				Point2f v = Point2f(X.at<double>(Point(0, 0)), X.at<double>(Point(0, 1)));
				if (v.x < neg4_x) neg4_x = v.x;
				if (v.y < neg4_y) neg4_y = v.y;

				if (v.x > max4_x) max4_x = v.x;
				if (v.y > max4_y) max4_y = v.y;
			}

			R.at<double>(Point(2, 0)) -= neg4_x;
			R.at<double>(Point(2, 1)) -= neg4_y;
			obj.R = R;
			obj.centre -= Point2f(neg4_x, neg4_y);

			if (center_max.x < obj.centre.x) center_max.x = obj.centre.x;
			if (center_max.y < obj.centre.y) center_max.y = obj.centre.y;

			max4_x -= neg4_x;
			max4_y -= neg4_y;

			if (max_size.width < max4_x) max_size.width = max4_x;
			if (max_size.height < max4_y) max_size.height = max4_y;
		}
	
		Point2f max_cen_diff = Point2f(0, 0);
		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			obj.centre_diff = center_max - obj.centre;
			if (obj.centre_diff.x > max_cen_diff.x) max_cen_diff.x = obj.centre_diff.x;
			if (obj.centre_diff.y > max_cen_diff.y) max_cen_diff.y = obj.centre_diff.y;
		}
		max_size.width += max_cen_diff.x;
		max_size.height += max_cen_diff.y;
		double len;
		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			Mat & obj_img = obj.obj_img;

			obj.R.at<double>(Point(2, 0)) += obj.centre_diff.x;
			obj.R.at<double>(Point(2, 1)) += obj.centre_diff.y;
			obj.centre += obj.centre_diff;
			warpAffine(obj_img, obj_img, obj.R, max_size, INTER_AREA);

			Point2f img_centre = Point2f(obj_img.cols / 2.0 , obj_img.rows / 2.0 );
			Point2f center_shift = img_centre - obj.centre;
			len = (norm(img_centre) + norm(center_shift)) * 2.0;
			Mat obj_temp = Mat::zeros(Size(len, len), CV_64FC1);

			Rect2f roi = Rect( Point2f(len/2.0 , len/2.0) - obj.centre, obj_img.size());
			obj_img.copyTo(obj_temp(roi));
			obj.obj_img = obj_temp;
		}
		
		Mat std_temp = Mat::zeros(Size(len, len), CV_64FC1);
		Rect2f std_roi = Rect(Point2f(len / 2.0, len / 2.0) - std.centre, std.obj_img.size());
		std.obj_img.copyTo(std_temp(std_roi));

		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			Mat & obj_img = obj.obj_img;
			Mat & tmp_img = std_temp;

			double err_gap = 0.05;
			double th_up = fabs(obj.ang_diff) * 2+ err_gap * 5 ;
			double th_down = -(fabs(obj.ang_diff) * 2 + err_gap * 5);

			for (; fabs(th_up - th_down) >= err_gap;)
			{			
				vector<double> new_th = bin_search(obj_img, tmp_img, th_down, th_up);
				if (new_th.size() < 2) continue;
				th_down = min(new_th[0], new_th[1]);
				th_up = max(new_th[0], new_th[1]);
			}
			obj.ang_diff = (th_down + th_up) / 2.0;

			size_t _n = os.first;
			obj_master_dir[_n] += obj.ang_diff;
		}

		for (auto & os : obj_st)
		{
			Obj_Set & obj = os.second;
			Mat & obj_img = obj.obj_img;

			//Mat A = match_features(obj_img , ml_obj::std.obj_img);		
			size_t _n = os.first;
					
			obj_master_dir[_n] += ml_obj::std.master_dir;
			float obj_dir = obj_master_dir[_n] * CV_PI / 180;
			Point2f master_vec_obj = Point2f(cos(obj_dir) , sin(obj_dir));
			obj_center[_n] += ml_obj::Oxy[_n];

			vector<Point2f> tripple = master_vec_to_tripple(obj_center[_n], master_vec_obj);
			vector<Point2f> _obj_points;
			for (auto & op : ml_obj::obj_points)
			{
				_obj_points.push_back(op * obj_scaler[_n]);
			}

			vector<Point2f> ow_points = Obj_to_Ow(_obj_points, tripple, ml_obj::vec_mat_H);

			
			map<size_t, bool> has_laser_out;
			for (; !has_laser_out[_n];)
			{
				vector<Point2f> p = ow_points;
				this_thread::sleep_for(chrono::milliseconds(1));
				Point2f bias_vec = Point2d(0, 0);

				
				map<size_t, Rect2d> tracking_obj = tracking_box;
				for (auto & it : tracking_obj)
				{
					size_t _tn = it.first;
					if (_n != _tn) continue;
					
					Rect2f t_roi = it.second;
					Rect2f roi_frame = Rect2f(t_roi.tl() / tracking_scaler, t_roi.br() / tracking_scaler) & Rect2f(Point2f(0, 0), frame.size());
					//if (!is_in_laser_region(roi_frame))
					//{
					//	this_thread::sleep_for(chrono::milliseconds(10));
					//	continue;
					//}

					diff_mxt.lock();
					ml_obj::mark = frame.clone();
					Mat & x = ml_obj::mark;
					bias_vec = get_centre(frame(roi_frame), roi_frame.tl()) - Point2d(obj_center[_n]);
					
					line(mark, (p[0] + bias_vec), (p[1] + bias_vec), Scalar(88), 1, LINE_AA);
					line(mark, (p[1] + bias_vec), (p[2] + bias_vec), Scalar(88), 1, LINE_AA);
					line(mark, (p[2] + bias_vec), (p[3] + bias_vec), Scalar(88), 1, LINE_AA);
					line(mark, (p[3] + bias_vec), (p[0] + bias_vec), Scalar(88), 1, LINE_AA);

					ml_obj::Ow_objs[_n] = ow_points;
					ml_obj::Oxy.erase(_n);
					ml_obj::objs.erase(_n);
					diff_mxt.unlock();

					has_laser_out[_n] = true;
					
					break;
				}
				
			}
			

			//ml_obj::obj_centre[_n] = obj_center[_n];

		}
		return;
	};

};

Obj_Set ml_obj::std = Obj_Set();
double ml_obj::lens_per_pixel = 1;
map<size_t, Mat> ml_obj::objs = map<size_t, Mat>();

map<size_t, diff_map> ml_obj::diff_set = map<size_t, diff_map>();

Mat ml_obj::mark = Mat();
Mat ml_obj::A = Mat();
double ml_obj::obj_area = 0;
map<size_t, Mat> ml_obj::objs_mark = map<size_t, Mat>();

map<size_t, Point2f> ml_obj::Oxy = map<size_t, Point2f>();
vector<Point2f> ml_obj::obj_points = vector<Point2f>();
vector<Mat> ml_obj::vec_mat_H = vector<Mat>();
map<size_t, vector<Point2f>> ml_obj::Ow_objs = map<size_t, vector<Point2f>>();
map<size_t, Point2d> ml_obj::obj_centre = map<size_t, Point2d>();


Obj_Set get_std_temp(Mat & _std, Point2f std_Oxy, vector<Point2f> Obj_point_in_Oc, double area_th = 0.0005)
{
	const size_t id_of_std = 0;
	Obj_Set std(id_of_std, _std.clone());
	Mat & std_img = std.obj_img;

	vector<vector<Point>> contours;
	cv::findContours(std_img, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	double area_total = 0;
	for (auto& c : contours)
	{
		area_total += moments(c, true).m00;
	}
	Point2f master_vec = Point2f(0, 0);
	vector<Point2f> avg_cen;
	for (auto & c : contours)
	{
		Moments Mpq = moments(c, true);
		double area_rate = Mpq.m00 / area_total;
		if (area_rate < area_th)
		{
			std_img(boundingRect(c)) = 0;
			continue;
		}
		else
		{
			ml_obj::obj_area += Mpq.m00;
		}
		Point2f centre = Point2f(static_cast<float>(Mpq.m10 / Mpq.m00), static_cast<float>(Mpq.m01 / Mpq.m00));
		avg_cen.push_back(centre * area_rate);
		master_vec += getOrientation(c, std_img, centre);
	}
	if (avg_cen.empty() || master_vec == Point2f(0, 0)) return Obj_Set();

	Point2f cen = Point2f(0, 0);
	for (size_t i = 0; i < avg_cen.size(); ++i)
	{
		cen += avg_cen[i];
	}

	double master_dir = atan2(master_vec.y, master_vec.x) * 180 / CV_PI;
	std.centre = cen;
	std.master_dir = master_dir;
	std.master_vec = master_vec;
	std.ang_diff = 0;

	vector<Point2f> master_tripple = master_vec_to_tripple(std.centre + std_Oxy, std.master_vec);

	//line(std_img, master_tripple[0] - std_Oxy, master_tripple[2] - std_Oxy, Scalar(128), 1, LINE_AA);
	//line(std_img, master_tripple[0] - std_Oxy, master_tripple[1] - std_Oxy, Scalar(196), 1, LINE_AA);
	Mat A = get_normed_affine_transform(master_tripple, false);
	if (!A.empty() && !Obj_point_in_Oc.empty())
	{
		ml_obj::A = get_normed_affine_transform(master_tripple);
		ml_obj::obj_points = re_mapping(Obj_point_in_Oc, A);
	}

	//line(std_img, ml_obj::obj_points[0], ml_obj::obj_points[1], Scalar(128), 1, LINE_AA);
	//line(std_img, ml_obj::obj_points[1], ml_obj::obj_points[2], Scalar(128), 1, LINE_AA);
	//line(std_img, ml_obj::obj_points[2], ml_obj::obj_points[3], Scalar(128), 1, LINE_AA);
	//line(std_img, ml_obj::obj_points[3], ml_obj::obj_points[0], Scalar(128), 1, LINE_AA);
	return std;
}

	//for (auto fi = 0; fi < fs.size(); ++fi)
	//{
	//	double tpf = (((double)getTickCount() - time3) / getTickFrequency()) * 1000;
	//	cout << "time per frame: " << tpf << " ms. " << fi <<endl;

	//	if (!tf.empty())
	//	{
	//		resize(tf, tf, Size(), 5, 5, INTER_AREA);
	//		imshow("tracker", tf);
	//		waitKey(1);
	//	}

	//	time3 = (double)getTickCount();
	//	//double time6 = (double)getTickCount();
	//	this_thread::sleep_for(chrono::milliseconds(1));
	//	frame = fs[fi];
	//	imwrite("erro.jpg", frame);
	//	static map<size_t, Mat> objs;
	//	static map<size_t, diff_map> diff_set;
	//	static ml_obj obj(objs, diff_set);
	//	double tracking_scaler = 0.1;
	//	resize(frame, tf, Size(), tracking_scaler, tracking_scaler);
	//	++d_frame;
	//	
	//	if (tracked_roi.empty() && mts[0].try_lock())
	//	{
	//		double time3 = (double)getTickCount();
	//		//cout << "matching roi fs : " << fi << endl;
	//		time = (double)getTickCount();
	//		tracking_all_lost = true;
	//		d_frame = 0;
	//		rois.clear();
	//		tracking_set.clear();
	//		objs.clear();
	//		
	//		rois = matching_template_rois(frame, temp);

	//		cout << "time detector: " << (((double)getTickCount() - time3) / getTickFrequency()) * 1000 << endl;

	//		time3 = (double)getTickCount();
	//		TrackerKCF::Params par;
	//		//r.compressed_size 
	//		for (int i = 0 ; i < rois.size(); ++i , ++n)
	//		{
	//			objs[n] = frame(rois[i]);
	//			
	//			
	//			tracking_set[n] = TrackerKCF::create();
	//			tracking_set[n]->init( tf , Rect2d(rois[i].tl() * tracking_scaler, rois[i].br() * tracking_scaler) );
	//		}

	//		cout << "time new tracker: " << (((double)getTickCount() - time3) / getTickFrequency()) * 1000 << endl;
	//	}
	//	map<size_t , Rect2d> tracking_box;
	//	tracked_roi.clear();
	//	
	//	//double time4 = (double)getTickCount();
	//	for (auto it = tracking_set.begin(); it != tracking_set.end(); )
	//	{
	//		size_t i = it->first;
	//		it->second->update(tf , tracking_box[i]);
	//		if (tracking_box[i].br().x > tf.cols)
	//		{
	//			//double time2 = (double)getTickCount();
	//			thread ts(&ml_obj::matching_template_accelerated, &obj, i);
	//			ts.detach();
	//			//ts.join();
	//			//time2 = (((double)getTickCount() - time2) / getTickFrequency()) * 1000;
	//			//cout << "time calc diff: " << time2 << endl;
	//			this_thread::sleep_for(chrono::milliseconds(1));
	//			tracking_set.erase(it++);
	//			tracking_box.erase(i);
	//			continue;
	//		}
	//		++it;
	//		tracked_roi.push_back(tracking_box[i]);
	//		rectangle(tf, tracking_box[i], Scalar(233, 0, 0), 1, 1);
	//	}
	//	//time4 = (((double)getTickCount() - time4) / getTickFrequency()) * 1000;
	//	//cout << "time tracking: " << time4 << endl;

	//	if (tracked_roi.empty() && objs.empty())
	//	{
	//		//d_frame /= rois.size();
	//		//d_frame /= 4;
	//		//fi += d_frame;
	//				
	//		if ( !mts[0].try_lock())
	//		{
	//			//cout << endl << "len :  ";
	//			//for (auto & ds : diff_set)
	//			//{
	//			//	cout << ds.second.max_scale_diff << "  ";
	//			//}
	//			time = (((double)getTickCount() - time) / getTickFrequency()) * 1000;
	//			cout << endl <<"time of once all : " << time / diff_set.size() << " ms. " << "  fs : " << d_frame / diff_set.size() << " objs : "<< diff_set.size() << endl;

	//			/*Mat res;
	//			//cout << endl << "iou :  ";
	//			for (auto & ds : diff_set)
	//			{
	//				//cout << ds.second.iou * 100 << "  ";
	//				Mat src = ds.second.diff.clone().t();
	//				if (src.cols * src.rows <= 1) continue;
	//				cv::resize(src, src, Size(diff_set.begin()->second.diff.size().width * scales, diff_set.begin()->second.diff.size().height *scales), 0, 0, INTER_AREA);
	//				src.convertTo(src, CV_8UC1);
	//				//string s = " len : " + to_string(ds.second.max_scale_diff);
	//				//set_lable(src , s , Point2f(0,0));
	//				res.push_back(src);
	//			}
	//			if (!res.empty()) imshow("diff", res.t());
	//				*/		
	//			//cout << "diff clear : " << diff_set.size() << endl;
	//			diff_set.clear();
	//			mts[0].unlock();
	//		}
	//		else 
	//		{
	//			mts[0].unlock();
	//		}
	//	}
	//	//Mat t = tf.clone();
	//	//resize(t, t, Size(), 5, 5, INTER_AREA);
	//	//imshow("tracker", t);
	//	//if (waitKey(1) == 27)continue;

	//	//time6 = (((double)getTickCount() - time6) / getTickFrequency()) * 1000;
	//	//cout << "time of per frame: " << time6 << endl;
	//}

	/*float scales = atof(scale.c_str());
	while (1)
	{
		double time = 0;
		static float degree = -1;
		degree += 1;
		degree = degree > 10 ? -10 : degree;
		Mat img_t = img.clone();
		Mat temp_t = temp.clone();
		//img_t = imrotate(img, degree);
		//temp_t = imrotate(temp, degree);
		//Mat temp_ts = imrotate(temp, -1);
		//imwrite("temp5.jpg",temp_ts);
		vector<diff_map> diff_set;
		time = (double)getTickCount();
		matching_template_accelerated(img_t, temp_t , diff_set);
		time = (((double)getTickCount() - time) / getTickFrequency()) * 1000 / diff_set.size();

		Mat res;
		cout << "iou :  ";
		for (size_t i = 0; i < diff_set.size(); ++i)
		{
			cout << diff_set[i].iou * 100 << "   ";
			Mat src = diff_set[i].diff.clone().t();
			if (src.cols * src.rows <= 1) continue;
			cv::resize(src, src, Size(), scales, scales, INTER_AREA);
			src.convertTo(src, CV_8UC1);
			//BrightnessAndContrastAuto(src, src);
			res.push_back(src);
		}
		cout << "  ( " << time << " ms.)  "  << degree << "·" << endl;
		cout << "len :  ";
		for (size_t i = 0; i < diff_set.size(); ++i)
		{
			cout << diff_set[i].max_scale_diff * df <<"   ";
		}
		cout << "df(mm) = " << df;
		cout << endl << endl;
		if (res.empty()) continue;
		imshow("diff", res.t());
		if ((waitKey(1) & 0xff) == 27) break;
	}*/


double get_scale_ds(vector<Mat> vec_mat,  double S = 1024 ,int N = 64 )
{
	double dx = 0;
	double dy = 0;
	int nx = 0;
	int ny = 0;

	for (int t = 1; t < N; ++t)
	{
		vector<Point2f> Oc_Points;
		vector<Point2f> Ow_Points;

		RNG rng((double)getTickCount());
		double y = rng.operator double() * S;

		double x = rng.operator double() * S;
		double z = rng.operator double() * S;

		Oc_Points.push_back(Point2f(x, y));
		Oc_Points.push_back(Point2f(z, y));
		Oc_to_Ow(Oc_Points, vec_mat, Ow_Points);
		if (Ow_Points.size() != 2) continue;
		dx += fabs(Ow_Points[0].x - Ow_Points[1].x) / fabs(z - x);
		++nx;
		Oc_Points.clear();
		Ow_Points.clear();

		x = rng.operator double() * S;
		y = rng.operator double() * S;
		z = rng.operator double() * S;

		Oc_Points.push_back(Point2f(x, y));
		Oc_Points.push_back(Point2f(x, z));
		Oc_to_Ow(Oc_Points, vec_mat, Ow_Points);
		if (Ow_Points.size() != 2) continue;
		dy += fabs(Ow_Points[0].y - Ow_Points[1].y) / fabs(z - y);
		++ny;
	}
	dx /= nx;
	dy /= ny;
	return (dx + dy) / 2.0;
}


//--------------------------------------------------------------------------------//

//find all related file  named  like"strkeyWords*" in dir 
// search dir with filter keyWords, store the filtered results in vImagePath
void getAllPicture(vector<string> &vImagePath,string strDir="../../data/tupian/", string strKeyWords="image")
{
	string strImagePath;
	string strSearchPath = strDir + strKeyWords + "*";

	intptr_t hFile = 0;
	struct _finddata_t fileinfo;

	if ((hFile = _findfirst(strSearchPath.c_str(), &fileinfo)) != -1)
	{
		do
		{
			strImagePath = strDir + fileinfo.name;
			vImagePath.push_back(strImagePath);
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}

}

void calibrate_camera(Pattern type, vector<Mat>vec_view, Size board_num, Size2f _real_size, vector<Mat> &vec_mat)
{
   vec_mat.clear();
   Size imageSize;	
   if (vec_view.size() == 0) return;
   vector<vector<Point3f> > Oc(vec_view.size());
   vector<vector<Point2f> > Ow(vec_view.size());
   cout << "calibration ..." << endl;
   for (int i = 0; i < vec_view.size(); ++i)
   {
	    Mat view = vec_view[i];
		
        Mat Oc_Points;
		bool found = false;
		switch (type)
		{
		case CHESSBOARD:
			found = findChessboardCorners(view, board_num, Oc_Points, CALIB_CB_FAST_CHECK | CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
			break;
		case CIRCLES_GRID:
		{
			SimpleBlobDetector::Params params;
			params.maxArea = 10000;
			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
			found = findCirclesGrid(view, board_num, Oc_Points, CALIB_CB_SYMMETRIC_GRID, detector);
			break;
		}
		case ASYMMETRIC_CIRCLES_GRID:
		{
			SimpleBlobDetector::Params params;
			params.maxArea = 10000;
			Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
			found = findCirclesGrid(view, board_num, Oc_Points, CALIB_CB_ASYMMETRIC_GRID, detector);
			//drawChessboardCorners(view, board_num, Mat(Oc_Points), found);
			break;
		}
		default:
			found = false;
			break;
		}
		CHECK_EXP_RETURN(!(found && (Oc_Points.channels() == 2)));
		cornerSubPix(view, Oc_Points, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));

        Mat Ow_Points;
		calcBoardCornerPositions(board_num, _real_size, Ow_Points, type);

		CHECK_EXP_RETURN(Ow_Points.empty() || Ow_Points.channels() != 3);


		vector<Mat> x_y;
		Mat z = Mat::zeros(Oc_Points.size(), CV_32FC1);
		split(Oc_Points, x_y);
		x_y.push_back(z);
		merge(x_y, z);
		Oc[i] = z;

		x_y.clear();
		split(Ow_Points, x_y);
		x_y.erase(--(x_y.end()));
		merge(x_y, z);
		Ow[i] = z;

		imageSize = view.size();
	}
        Mat cameraMatrix, distCoeffs;
		vector<Mat> rvecs, tvecs;
		int flags = CALIB_FIX_PRINCIPAL_POINT;// | CALIB_ZERO_TANGENT_DIST | CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3;
   		double rms = calibrateCamera(Oc, Ow, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags);
		CHECK_EXP_RETURN(rms > 1);		
		
		vec_mat.push_back(rvecs[0]);
		vec_mat.push_back(tvecs[0]);
		vec_mat.push_back(cameraMatrix);
		vec_mat.push_back(distCoeffs);


	string outPutFile = "../../data/camera.xml";
	cv::FileStorage fs(outPutFile, cv::FileStorage::WRITE);
#ifdef _DEBUG
	fs << "image_width" << imageSize.width << "image_height" << imageSize.height << "camera_matrix" << cameraMatrix << "distortion_coefficients"
		<< distCoeffs;
	fs << "Reprojection-error" << rms;

    fs << "rvecs-0"  << rvecs[0];

    fs << "tvecs-0" << tvecs[0];
#endif
	fs << "vec_mat" << vec_mat;
   
	fs.release();


	return;
}

class CWindowsCamera
{
public:
	int m_nRet = MV_OK;
	void* m_handle = NULL;
	unsigned int m_nPayloadSize = 0;
	unsigned int m_nFormat = 0;

	CWindowsCamera();
	~CWindowsCamera();
	void WaitForKeyPress(void);
	int RGB2BGR(unsigned char* pRgbData, unsigned int nWidth, unsigned int nHeight);
	bool PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo);
	bool Convert2Mat(MV_FRAME_OUT_INFO_EX* pstImageInfo, unsigned char* pData, cv::Mat& srcImage);
	
	bool setCamera(void * &handle, unsigned int & nPayloadSize, unsigned int &nFormat);
	//if success return 1;else 0
	bool myCapture(void * handle, int nRet, unsigned int nPayloadSize, int nFormat, cv::Mat &srcImage);
	bool myReleaseCamera(void * handle, int nRet);

	bool setCamera();
	bool myCapture(cv::Mat &srcImage);

};

CWindowsCamera::CWindowsCamera()
{

}
CWindowsCamera::~CWindowsCamera()
{
	myReleaseCamera(m_handle, m_nRet);

}

bool CWindowsCamera::setCamera()
{
	return setCamera(m_handle,  m_nPayloadSize, m_nFormat);
}

bool CWindowsCamera::myCapture(cv::Mat &srcImage)
{
	return myCapture(m_handle, m_nRet, m_nPayloadSize, m_nFormat, srcImage);
}

// Wait for key press
void CWindowsCamera::WaitForKeyPress(void)
{
	while (!_kbhit())
	{
		Sleep(10);
	}
	_getch();
}

// print the discovered devices information to user
bool CWindowsCamera::PrintDeviceInfo(MV_CC_DEVICE_INFO* pstMVDevInfo)
{
	if (NULL == pstMVDevInfo)
	{
		printf("The Pointer of pstMVDevInfo is NULL!\n");
		return false;
	}
	if (pstMVDevInfo->nTLayerType == MV_GIGE_DEVICE)
	{
		int nIp1 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24);
		int nIp2 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16);
		int nIp3 = ((pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8);
		int nIp4 = (pstMVDevInfo->SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff);

		// print current ip and user defined name
		//printf("CurrentIp: %d.%d.%d.%d\n", nIp1, nIp2, nIp3, nIp4);
		//printf("UserDefinedName: %s\n\n", pstMVDevInfo->SpecialInfo.stGigEInfo.chUserDefinedName);
	}
	else if (pstMVDevInfo->nTLayerType == MV_USB_DEVICE)
	{
		printf("UserDefinedName: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chUserDefinedName);
		printf("Serial Number: %s\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.chSerialNumber);
		printf("Device Number: %d\n\n", pstMVDevInfo->SpecialInfo.stUsb3VInfo.nDeviceNumber);
	}
	else
	{
		printf("Not support.\n");
	}

	return true;
}

int CWindowsCamera::RGB2BGR(unsigned char* pRgbData, unsigned int nWidth, unsigned int nHeight)
{
	if (NULL == pRgbData)
	{
		return MV_E_PARAMETER;
	}

	for (unsigned int j = 0; j < nHeight; j++)
	{
		for (unsigned int i = 0; i < nWidth; i++)
		{
			unsigned char red = pRgbData[j * (nWidth * 3) + i * 3];
			pRgbData[j * (nWidth * 3) + i * 3] = pRgbData[j * (nWidth * 3) + i * 3 + 2];
			pRgbData[j * (nWidth * 3) + i * 3 + 2] = red;
		}
	}

	return MV_OK;
}

// convert data stream in Mat format
bool CWindowsCamera::Convert2Mat(MV_FRAME_OUT_INFO_EX* pstImageInfo, unsigned char* pData, cv::Mat& outputImage)
{
	cv::Mat srcImage;
	//std::cout << pstImageInfo->enPixelType << std::endl;
	if (pstImageInfo->enPixelType == PixelType_Gvsp_Mono8)
	{
		srcImage = cv::Mat(pstImageInfo->nHeight, pstImageInfo->nWidth, CV_8UC1, pData);
	}
	else if (pstImageInfo->enPixelType == PixelType_Gvsp_RGB8_Packed)
	{
		RGB2BGR(pData, pstImageInfo->nWidth, pstImageInfo->nHeight);
		srcImage = cv::Mat(pstImageInfo->nHeight, pstImageInfo->nWidth, CV_8UC3, pData);
	}
	else
	{
		printf("unsupported pixel format\n");
		return false;
	}

	if (NULL == srcImage.data)
	{
		return false;
	}
	else
	{
		outputImage = srcImage;
	}

	//	//save converted image in a local file
	//	try {
	//#if defined (VC9_COMPILE)
	//		cvSaveImage("MatImage.bmp", &(IplImage(srcImage)));
	//#else
	//		cv::imwrite("MatImage.bmp", srcImage);
	//#endif
	//	}
	//	catch (cv::Exception& ex) {
	//		fprintf(stderr, "Exception saving image to bmp format: %s\n", ex.what());
	//	}


		//srcImage.release();

	return true;
}


bool CWindowsCamera::setCamera(void * &handle, unsigned int &nPayloadSize, unsigned int &nFormat)
{
	//Read the video stream directly from the camer

	if (m_handle != NULL)
	{
		return 1;
	}

	int nRet = MV_OK;
	// Enum device
	MV_CC_DEVICE_INFO_LIST stDeviceList;
	memset(&stDeviceList, 0, sizeof(MV_CC_DEVICE_INFO_LIST));
	nRet = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &stDeviceList);
	if (MV_OK != nRet)
	{
		printf("Enum Devices fail! nRet [0x%x]\n", nRet);
		return 0;
	}

	if (stDeviceList.nDeviceNum > 0)
	{
		for (unsigned int i = 0; i < stDeviceList.nDeviceNum; i++)
		{
			//printf("[device %d]:\n", i);
			MV_CC_DEVICE_INFO* pDeviceInfo = stDeviceList.pDeviceInfo[i];
			if (NULL == pDeviceInfo)
			{
				break;
			}
			PrintDeviceInfo(pDeviceInfo);
		}
	}
	else
	{
		return 0;
	}

	// input the format to convert
	//printf("[0] OpenCV_Mat\n");
	//printf("[1] OpenCV_IplImage\n");
	//printf("Please Input Format to convert:");

	//scanf_s("%d", &nFormat);
	if (nFormat >= 2)
	{
		printf("Input error!\n");
		return 0;
	}

	// select device to connect
	//printf("Please Input camera index:");
	unsigned int nIndex = 0;
	//scanf_s("%d", &nIndex);
	if (nIndex >= stDeviceList.nDeviceNum)
	{
		printf("Input error!\n");
		return 0;
	}

	// Select device and create handle
	nRet = MV_CC_CreateHandle(&handle, stDeviceList.pDeviceInfo[nIndex]);
	if (MV_OK != nRet)
	{
		printf("Create Handle fail! nRet [0x%x]\n", nRet);
		return 0;
	}

	// open device
	nRet = MV_CC_OpenDevice(handle);
	if (MV_OK != nRet)
	{
		printf("Open Device fail! nRet [0x%x]\n", nRet);
		return 0;
	}
	// Detection network optimal package size(It only works for the GigE camera)
	if (stDeviceList.pDeviceInfo[nIndex]->nTLayerType == MV_GIGE_DEVICE)
	{
		int nPacketSize = MV_CC_GetOptimalPacketSize(handle);
		if (nPacketSize > 0)
		{
			nRet = MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
			if (nRet != MV_OK)
			{
				printf("Warning: Set Packet Size fail nRet [0x%x]!", nRet);
			}
		}
		else
		{
			printf("Warning: Get Packet Size fail nRet [0x%x]!", nPacketSize);
		}
	}
	// Set trigger mode as off
	nRet = MV_CC_SetEnumValue(handle, "TriggerMode", 0);
	if (MV_OK != nRet)
	{
		printf("Set Trigger Mode fail! nRet [0x%x]\n", nRet);
		return 0;
	}

	// Get payload size
	MVCC_INTVALUE stParam;
	memset(&stParam, 0, sizeof(MVCC_INTVALUE));
	nRet = MV_CC_GetIntValue(handle, "PayloadSize", &stParam);
	if (MV_OK != nRet)
	{
		printf("Get PayloadSize fail! nRet [0x%x]\n", nRet);
		return 0;
	}
	nPayloadSize = stParam.nCurValue;

	// Start grab image
	nRet = MV_CC_StartGrabbing(handle);
	if (MV_OK != nRet)
	{
		printf("Start Grabbing fail! nRet [0x%x]\n", nRet);
		return 0;
	}

	return 1;
}

bool CWindowsCamera::myCapture(void * handle, int nRet, unsigned int nPayloadSize, int nFormat, Mat &matImage)
{
	MV_FRAME_OUT_INFO_EX stImageInfo = { 0 };
	memset(&stImageInfo, 0, sizeof(MV_FRAME_OUT_INFO_EX));
	unsigned char* pData = (unsigned char*)malloc(sizeof(unsigned char) * (nPayloadSize));
	if (pData == NULL)
	{
		printf("Allocate memory failed.\n");
		return 0;
	}

	// get one frame from camera with timeout=1000ms
	nRet = MV_CC_GetOneFrameTimeout(handle, pData, nPayloadSize, &stImageInfo, 1000);
	if (nRet == MV_OK)
	{
		//printf("Get One Frame: Width[%d], Height[%d], nFrameNum[%d]\n",
		//	stImageInfo.nWidth, stImageInfo.nHeight, stImageInfo.nFrameNum);
	}
	else
	{
		printf("No data[0x%x]\n", nRet);
		free(pData);
		pData = NULL;
		return 0;
	}

	// data transform as mat
	cv::Mat srcImage;//original Image
	bool bConvertRet = false;
	if (0 == nFormat)
	{
		bConvertRet = Convert2Mat(&stImageInfo, pData, srcImage);
	}
	matImage = srcImage;
	return 1;
}

bool CWindowsCamera::myReleaseCamera(void * handle, int nRet)
{
	// Stop grab image
	nRet = MV_CC_StopGrabbing(handle);
	if (MV_OK != nRet)
	{
		//printf("Stop Grabbing fail! nRet [0x%x]\n", nRet);
		return 0;
	}

	// Close device
	nRet = MV_CC_CloseDevice(handle);
	if (MV_OK != nRet)
	{
		printf("ClosDevice fail! nRet [0x%x]\n", nRet);
		return 0;
	}

	// Destroy handle
	nRet = MV_CC_DestroyHandle(handle);
	if (MV_OK != nRet)
	{
		printf("Destroy Handle fail! nRet [0x%x]\n", nRet);
		return 0;
	}

	if (nRet != MV_OK)
	{
		if (handle != NULL)
		{
			MV_CC_DestroyHandle(handle);
			handle = NULL;
		}
	}
	return 1;

}

void CreateDir(const char *dir)
{
	int m = 0, n=0;
	string str1, str2;
	str1 = dir;
	m = str1.find('/');

	while (m >= 0)
	{
        
		str2 = str2+str1.substr(0, m) + '/';
		n = _access(str2.c_str(), 0);
		
		if (n == -1)
		{
			_mkdir(str2.c_str());
		}
        str1 = str1.substr(m + 1, str1.size());
		m = str1.find('/');
	}
	str1 = str1.substr(m + 1, str1.size());
	str2 = str2 + str1.substr(0, m) + '/';
	_mkdir(str2.c_str());
}



BOOL removeDir(string path)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;  
	string p;

	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))    
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					removeDir(p.assign(path).append("/").append(fileinfo.name));     
					_rmdir(fileinfo.name);
				}
			}
			else    
			{
				string strFile = p.assign(path).append("/").append(fileinfo.name);
				DeleteFile(strFile.c_str()); 
			}
		} while (_findnext(hFile, &fileinfo) == 0);  
		_findclose(hFile);
	}
	return 0;
}

//strDir is the path to save image,strKeywords is the rule to name 
BOOL myCapture(vector<Mat> &vec_view, string strDir = "../../data/tupian", string strKeyword = "image")
{
	int iNumber = 0;
	CWindowsCamera myCamera;
	if (myCamera.setCamera(myCamera.m_handle, myCamera.m_nPayloadSize, myCamera.m_nFormat))
	{
		cout << "press Enter to save image,Esc to escape" << endl;
		do
		{
			Mat srcImage;
			myCamera.myCapture(myCamera.m_handle, myCamera.m_nRet, myCamera.m_nPayloadSize, myCamera.m_nFormat, srcImage);
			if (srcImage.empty())
			{
				cout << "capture fail" << endl;
				break;
			}

			cv::namedWindow("image", WINDOW_GUI_NORMAL);
			cv::imshow("image", srcImage);

			//input Enter to save image
			int flag = waitKey(10);
			if (flag == 13)
			{
#ifdef _DEBUG
				string name = strDir + "/" + strKeyword + to_string(iNumber) + ".jpg";
				imwrite(name, srcImage);
#endif
				cout << "save image" + to_string(iNumber) + ".jpg success" << endl;
				vec_view.push_back(srcImage);
				++iNumber;
				continue;
			}
			//input Esc to break
			if (flag == 27)
			{
				if (vec_view.size() == 0)
				{
					cout << "number of image can not be zero,please press Enter to capture" << endl;
					continue;
				}
				break;
			}


		} while (MV_OK == myCamera.m_nRet);
	}
	else
	{
		cout << "set camera fail" << endl;	
		return 0;
	}

	return 1;
}

BOOL getCalibrateImage(vector<Mat> & vec_view,string strDir="../../data/tupian")
{

    //DeleteFile("../../data/camera.xml");

#ifdef _DEBUG
	if (_access(strDir.c_str(), 0) == 0)
	{
		int flag = removeDir(strDir.c_str());

		if (flag == 0)
		{
			cout << "delete original calibration images successfully" << endl;
		}

	}
	CreateDir(strDir.c_str());
#endif
	if (!myCapture(vec_view))
	{
		return 0;
	};
	return 1;

}

int loadCalib(string file, vector<Mat>& vec_mat)
{
	vec_mat.clear();
	cv::FileStorage fs(file, cv::FileStorage::READ);

	// Example of loading these matrices back in
	if (!fs.open(file, cv::FileStorage::READ))
	{
		cout << "load calibration matrix fail" << endl;
		return 0;
	}


	cv::Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
	fs["vec_mat"] >> vec_mat;
	

	return 1;
}

BOOL cameraDetect()
{
    CWindowsCamera myCameraTest;
	if (myCameraTest.setCamera(myCameraTest.m_handle, myCameraTest.m_nPayloadSize, myCameraTest.m_nFormat))
	{
		cout << "Found Camera " << endl;
		return 1;
	}
	else
	{
		cout << "Find No Camera " << endl;
		return 0;
	}
}

bool detect_camera(CWindowsCamera &myCameraTest)
{
	if (myCameraTest.setCamera(myCameraTest.m_handle, myCameraTest.m_nPayloadSize, myCameraTest.m_nFormat))
	{
		return true;
	}
	else
	{
		return false;
	}
}

std::string getCurrentSystemTime()
{
	auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	struct tm *ptm = localtime(&tt);
	char date[60] = { 0 };
	sprintf(date, "%d-%02d-%02d-%02d.%02d.%02d",
		(int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
		(int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
	return std::string(date);
}

void calibra2(vector<Mat> &vec_mat,Pattern type=CHESSBOARD,Size board_num=Size(11,8),Size2f real_size=Size2f(3.0,3.0))
{
	string strCamera = "../../data/camera.xml";
	if (_access(strCamera.c_str(), 0) == 0)
	{
		cout << "camera.xml exist" << endl;
	}
	else
	{
		cout<< "camera.xml does not exist" << endl;
	}


	vector<Mat> vec_view;        //image captured
#ifdef _DEBUG
    string strDir = "../../data/tupian";
	vector<string> vImagePath;
	string strKeyWords = "image";
#endif
	if (cameraDetect())
	{
		const double wait_time = 3;
		int ch = 0;
	    cout << "press SPACCE to Recalibrate Camera ?  or Enter to skip. Waiting for " << wait_time << "s"  << endl;
		std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - startTime);
		for (; time_span.count() < wait_time;)
		{			
			if (_kbhit()) //如果有按键按下，则_kbhit()函数返回真
			{
                 ch = _getch();//使用_getch()函数获取按下的键值
			}
				
		    if(ch == 32)
			{
               if (getCalibrateImage(vec_view))
			   {           
			     calibrate_camera(type, vec_view, board_num, real_size, vec_mat);
				 break;
			   }
			   else
		 	   {
				cout << "open camera fail,automatically use original camera.xml" << endl;
				loadCalib("../../data/camera.xml", vec_mat);
				break;
		       }	
		    }
			else if(ch == 13)
			{
				break;
			}
			std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
			time_span = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);

		}
		if(vec_mat.size()==0)
	    {
               cout << "automatically use original camera.xml" << endl;
		       loadCalib("../../data/camera.xml", vec_mat);
		
		}
	
	}
	else
	{
		cout << "automatically use original camera.xml" << endl;
		loadCalib("../../data/camera.xml", vec_mat);
	}
}



Mat & get_next_frame(string src_name = string() , double f_scale = 1  , long t_delay = 1 )
{
	typedef enum { frome_none = 0, from_camera, from_vedio, from_img, type_max }src_frame_type;
	static src_frame_type src_type = frome_none;

	static Mat next_frame = Mat();
	static CWindowsCamera Camera;
	static VideoCapture VC;
	static vector<Mat> fs;
	static int nf = 0;
	static double frame_scaler = 1;
	static long time_delay = 1;
	static string src_path = string();
	if (!src_name.empty())
	{
		if (src_name != src_path) {
			src_path = src_name;
			src_type = frome_none;
		}
		frame_scaler = f_scale;
		time_delay = t_delay;
	}
	if (src_type == frome_none)
	{
		if (detect_camera(Camera))
		{
			src_type = from_camera;
#ifdef _DEBUG
			cout << "found src frame frome Camera " <<endl;
#endif
		}
		else if (VC.open(src_name) && VC.get(CAP_PROP_FRAME_COUNT) > 4)
		{
			src_type = from_vedio;
#ifdef _DEBUG
			cout << "found src frame frome Vedio "<< endl;
#endif
		}
		else if (!imread(src_name).empty())
		{		
			VC = gen_video_from_img(src_name);
			if (VC.isOpened()) 
			{
				src_type = from_img;
			}
#ifdef _DEBUG
			cout << "found src frame frome Img " << endl;
#endif
		}
		else
		{
			src_type = frome_none;
#ifdef _DEBUG
			bool isopen = VC.open(src_name);
			auto cnt = VC.get(CAP_PROP_FRAME_COUNT);
			cout << "src frame type_none error " << endl;
#endif
		}
	}
	switch (src_type)
	{
		case from_camera:
		{
			Camera.myCapture(next_frame);
			break;
		}
		case from_vedio:
		{
			VC >> next_frame;
			break;
		}
		case from_img:
		{
			VC >> next_frame;
			break;
		}
		default: break;
	}

	if (frame_scaler > 0 && frame_scaler < 1 && !next_frame.empty())
	{
		resize(next_frame, next_frame, Size(), frame_scaler, frame_scaler, INTER_AREA);
	}
	if (next_frame.channels() != 1 && !next_frame.empty())
	{
		cvtColor(next_frame, next_frame, CV_BGR2GRAY);
	}
	if (next_frame.type() != CV_8UC1) next_frame.convertTo(next_frame, CV_8UC1);
	next_frame = 255 - next_frame;
	this_thread::sleep_for(chrono::milliseconds(time_delay));
	return next_frame;
}


Mat rechoose_tmp(string img_path,string temp_path, Point2f & std_oxy , vector<Point2f> & Obj_point_in_Oc  ,double scaler = 1.0)
{
	Mat temp = imread(temp_path, IMREAD_GRAYSCALE);
	Rect roi = Rect(0,0,temp.cols , temp.rows);
	Mat selected_frame = temp.clone();
	int waitime = 3;
	cout << "press SPACE re_select template?  or Enter to skip." << "  Waiting for ..." << waitime <<"s" << endl;
	char ch ;
	std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now()-startTime);
	string win = "press SPACE confirm !";
	for(;time_span.count()<waitime;)
	{
		if (_kbhit()) //如果有按键按下，则_kbhit()函数返回真
		{
			ch = _getch();//使用_getch()函数获取按下的键值
			if (ch == 'y' || ch == 32)
			{
				for (Mat& frame = get_next_frame(img_path, scaler); true; frame = get_next_frame())
				{
					if ((waitKey(1) == 32) && !frame.empty())
					{
						
						namedWindow(win);
						setWindowProperty(win , WND_PROP_AUTOSIZE, WINDOW_KEEPRATIO);
						roi = selectROI(win, frame, false);

						selected_frame = frame;
						std_oxy = roi.tl();

						if (roi.empty())continue;
						temp = frame(roi).clone();
						imwrite(temp_path, temp);
						//temp = imread(temp_path, IMREAD_GRAYSCALE);
						if (temp.type() != CV_8UC1)temp.convertTo(temp, CV_8UC1);
						break;
					}

					if(frame.empty())
					{
						continue;
					}
					else 
					{
						//namedWindow(win, WINDOW_KEEPRATIO);
						imshow(win, frame);
						if ((waitKey(1)) == 27) return temp;
					}
				}
				cv::destroyAllWindows();
				break;
			}
			else if(ch=='n' || ch == 13)
			{
 				break;
			}
		}
		std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
		time_span = std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
	}
	if (temp.empty()) {
		cout <<"\nThere is no template image in"<<temp_path<<". "
			"Please check whether your path setting is correct or re-select the template image."
			"\nIf you need to re-select the template image, please press the SPACE key to select the template in the display window."
			"\nIf there is no object in the display window, press the ESC key to exit directly." << endl;
		for (Mat& frame = get_next_frame(img_path, scaler); true; frame = get_next_frame())
		{
			if ((waitKey(1) == 32) && !frame.empty())
			{
				Rect roi;
				namedWindow(win, WINDOW_AUTOSIZE);
				roi = selectROI(win, frame, false);
				if (roi.empty())continue;
				temp = frame(roi).clone();
				imwrite(temp_path, temp);
				if (temp.type() != CV_8UC1)temp.convertTo(temp, CV_8UC1);
				break;
			}
			if (frame.empty())continue;
			else {
				namedWindow(win, WINDOW_AUTOSIZE);
				imshow(win, frame);
				if ((waitKey(10)) == 27) break;
			}
		}
		cv::destroyAllWindows();
	}

	selected_frame = 255 - selected_frame;
	rectangle(selected_frame, roi, Scalar(0, 0, 0), 1, 1);
	auto roi_laser_gragh = selectROI(win, selected_frame, false);
	Obj_point_in_Oc.push_back(roi_laser_gragh.tl());
	Obj_point_in_Oc.push_back(Point2f(roi_laser_gragh.br().x, roi_laser_gragh.tl().y));
	Obj_point_in_Oc.push_back(roi_laser_gragh.br());
	Obj_point_in_Oc.push_back(Point2f(roi_laser_gragh.tl().x, roi_laser_gragh.br().y));

	img_pre_processing(temp);
	return temp;
}

float IOU(const cv::Rect2f & r1, const cv::Rect2f & r2)
{
	return (r1 & r2).area() / (r1 | r2).area();
}
Mat get_normed_center_affine_transform(const vector<Point2f>& src_tri, bool obj_to_oc = true)
{
	Mat_<double> A;
	double scale_x = norm(src_tri[1] - src_tri[0]);
	double scale_y = norm(src_tri[1] - src_tri[2]);
	vector<Point2f> dst_trip(3); //refer to obj_tripple

	dst_trip[0] = Point2f(ml_obj::std.centre.x + scale_x, ml_obj::std.centre.y);
	dst_trip[1] = Point2f(ml_obj::std.centre.x, ml_obj::std.centre.y);
	dst_trip[2] = Point2f(ml_obj::std.centre.x, ml_obj::std.centre.y + scale_y);
	if (obj_to_oc)
	{
		A = getAffineTransform(dst_trip, src_tri);
	}
	else
	{
		A = getAffineTransform(src_tri, dst_trip);
		//Mat A_inv = getAffineTransform(src_tri, dst_trip);
		//vector<Point2f> dst = re_mapping(src_tri ,A_inv);
	}

	return A;
}




void obj_scale_detect(string img_path , string temp_path , double scaler = 1.0 , vector<Mat> vec_mat = vector<Mat>() )
{
	Point2f std_Oxy = Point2f(0,0);
	vector<Point2f> Obj_point_in_Oc = vector<Point2f>();
	 Mat temp = rechoose_tmp(img_path , temp_path , std_Oxy, Obj_point_in_Oc , scaler);
	 if (temp.empty() || Obj_point_in_Oc.empty())
	 {
		 cout << "error : temp empty !" << endl;
		 return;
	 }
	ml_obj::std = get_std_temp(temp ,std_Oxy, Obj_point_in_Oc);
	ml_obj::vec_mat_H = vec_mat;

	
	
	static Mat tf;
	static Mat frames;
	size_t n = 1;
	double tracking_scaler = 0.2;
	map<size_t, Rect2d> _tracking_box;
	map<size_t, Ptr<Tracker>> tracking_set;


	const int f_error_times = 5;
	int error_times = f_error_times;	
	for (frames = get_next_frame(img_path, scaler , 1); error_times; frames = get_next_frame())
	{
		if (frames.empty())
		{
			--error_times;
			continue;
		}
		error_times = f_error_times;

		static ml_obj obj(frames,tracking_scaler, _tracking_box);

		if (!tf.empty())
		{
			Mat t = tf.clone();
			resize(t, t, Size(), 2, 2, INTER_AREA);
			imshow("tracker", t);
			if (waitKey(1) == 27) break;
		}
		cv::resize(frames, tf, Size(), tracking_scaler, tracking_scaler);

		vector<Rect2d> new_rois;
		if (true)
		{
			img_pre_processing(frames);
			vector<Rect2d> rois = matching_template_rois(frames, temp);

			const double iou_th = 0.3;
			for (auto & roi : rois)
			{
				double max_iou = 0;
				Rect2d r = Rect2d(roi.tl() * tracking_scaler, roi.br() * tracking_scaler);
				multiset<double, greater<double>> max_roi;
				for (auto & t_box : _tracking_box)
				{
					max_roi.insert((r & t_box.second).area() / (r | t_box.second).area());
				}
				if (!max_roi.empty())  max_iou = *max_roi.begin();
				if (max_iou < iou_th)  new_rois.push_back(roi);
			}
		}
		if (!new_rois.empty())
		{
			//ml_obj::objs.clear();
			if (mts_master.try_lock())
			{
				for (int i = 0; i < new_rois.size(); ++i, ++n)
				{
					tracking_set[n] = TrackerKCF::create();
					tracking_set[n]->init(tf, Rect2d(new_rois[i].tl() * tracking_scaler, new_rois[i].br() * tracking_scaler));

					ml_obj::Oxy[n] = new_rois[i].tl();
					ml_obj::objs[n] = frames(new_rois[i]);				
					thread ts(&ml_obj::matching_template_accelerated, &obj, n);
					//ts.join();
					ts.detach();				
				}
			}
			else
			{
				this_thread::sleep_for(chrono::milliseconds(1));
			}
		}
		else
		{
			// skip frames 
		}

		for (auto it = tracking_set.begin(); it != tracking_set.end(); )
		{
			static double iot_th = 0.8;
			size_t n = it->first;

			bool tracking_lost = !(it->second->update(tf, _tracking_box[n]));
			rectangle(tf, _tracking_box[n], Scalar(233, 0, 0), 1, 1);

			double iot = (_tracking_box[n] & Rect2d(Point2d(0, 0), Size(tf.cols, tf.rows))).area() / _tracking_box[n].area();
			if( iot < iot_th || tracking_lost)
			{
				tracking_set.erase(it++);
				_tracking_box.erase(n);
				continue;
			}
			++it;
		}

		bool diff_over =  ml_obj::objs.empty();
		if (diff_over)
		{
			if (!mts_master.try_lock())
			{
#ifdef RES_OUT
					cout << endl << "len: "; //显示测量的最大差异尺寸
					for (auto & ds : ml_obj::diff_set)
					{
						cout << ds.second.max_scale_diff << "  ";
					}
					cout <<".mm per pixel: " << ml_obj::lens_per_pixel << endl;

					Mat res;
					cout << endl << "iou: "; //显示IOU
					for (auto & ds : ml_obj::diff_set)
					{
						cout << ds.second.iou * 100 << "  ";
						Mat src = ds.second.diff.clone().t();
						if (src.cols * src.rows <= 1) continue;
						cv::resize(src, src, Size(ml_obj::diff_set.begin()->second.diff.size().width * scaler, ml_obj::diff_set.begin()->second.diff.size().height *scaler), 0, 0, INTER_AREA);
						src.convertTo(src, CV_8UC1);
						res.push_back(src);
					}
					cout << endl << endl;
					if (!res.empty()) imshow("diff", res.t()); //显示差分图
						
#endif
															   
					for (auto ow_obj = ml_obj::Ow_objs.begin(); ow_obj != ml_obj::Ow_objs.end();)
					{
						/*size_t _n = ow_obj->first;
						
						vector<Point2f> p = ow_obj.second;
						//Rect2d ow_rect = Rect2d(p[0] , p[2]);
						//Rect2d ow_rect2 = Rect2d(p[0], p[2]);
						Point2f bias_vec = Point2d(0, 0);
						for (auto it = tracking_box.begin(); it != tracking_box.end(); ++it)
						{
							size_t _tn = it->first;
							if (_n != _tn) continue;
							Rect2f t_roi = it->second;
							Rect2f roi_frame = Rect2f(t_roi.tl() / tracking_scaler, t_roi.br() / tracking_scaler) & Rect2f(Point2f(0,0), frame.size());
							//rectangle(frame, roi_frame, Scalar(255), 1, LINE_4);
							bias_vec = get_centre(frame(roi_frame) , roi_frame.tl()) - ml_obj::obj_centre[_n];
							//ow_rect += bias_vec;
							//ow_rect = Rect2d(ow_rect.tl()*tracking_scaler, ow_rect.br()*tracking_scaler);
							//if (!is_obj_in_laser_region) continue;
							line(tf, (p[0]+ bias_vec)*tracking_scaler, (p[1] + bias_vec)*tracking_scaler, Scalar(128), 1, LINE_AA);
							line(tf, (p[1] + bias_vec)*tracking_scaler, (p[2] + bias_vec)*tracking_scaler, Scalar(128), 1, LINE_AA);
							line(tf, (p[2] + bias_vec)*tracking_scaler, (p[3] + bias_vec)*tracking_scaler, Scalar(128), 1, LINE_AA);
							line(tf, (p[3] + bias_vec)*tracking_scaler, (p[0] + bias_vec)*tracking_scaler, Scalar(128), 1, LINE_AA);
							*/	
							//rectangle(tf, ow_rect, Scalar(32), 1, LINE_4);
							if (!tf.empty())
							{
								Mat t = ml_obj::mark.clone();
								resize(t, t, Size(), 2 * tracking_scaler, 2 * tracking_scaler, INTER_AREA);
								imshow("tracker", t);
								if (waitKey(50) == 27) break;
							}
							ml_obj::Ow_objs.erase(ow_obj++);
					}			
					mts_master.unlock();
			}
			else
			{
				mts_master.unlock();
			}
		}

	}
	mts_master.try_lock();
	mts_master.unlock();
}

int main(int argc, char** argv)
{
	string img_no, scale;
	if (argc <= 1) 
	{
		img_no = string("1");
		scale = string("0.5");
	}
	if (argc >= 3)
	{
		img_no = argv[1];
		scale = argv[2];
	}
	else if (argc == 2)
	{
		img_no = argv[1];
		scale = string("0.5");
	};

	Pattern type = CHESSBOARD;
	Size board_num = Size(11, 8);
	Size2f real_size = Size2f(3.0, 3.0);
	static vector<Mat> vec_mat;
	calibra2(vec_mat,type,board_num ,real_size);

    CHECK_EXP_RETURN(vec_mat.size() != 4, SYS_CALIBRATE_CONFIG_ERROR);
	cout << "calibrate camera succeed."<< endl << endl;

	string temp_name = string("../../data/temp") + img_no + ".jpg";
	string img_name = string("../../data/obj") + img_no + ".avi";
	//string mark_name = string("../../data/mark") + ".avi";
    obj_scale_detect(img_name, temp_name , atof(scale.c_str()) , vec_mat);
	system("pause");
}


	/*{
		Mat view = imread("C:\\Users\\EDZ\\Desktop\\opencv_cablication\\chessboard0.png", IMREAD_GRAYSCALE);
		Size board_num(7, 7);
		Size2f real_size(62.93, 65.2);
		view_config view_cfg = view_config(view, CHESSBOARD, real_size, board_num);
		img_config img_cfg = img_config();
		err_type err_code = Laser_Printer_System(img_cfg, view_cfg);
		CHECK_EXP_RETURN(err_code != CALIBRATE_SUCCESS, err_code);
	}

	{
		Mat img = imread("C:\\Users\\EDZ\\Desktop\\opencv_cablication\\Image_20190613164200.jpg", IMREAD_GRAYSCALE);
		Mat temp = imread("C:\\Users\\EDZ\\Desktop\\opencv_cablication\\ss.png", IMREAD_GRAYSCALE);

		vector<Point2f> obj_points = matching_template_accelerated_knn(img, temp);
		img_config img_cfg = img_config(img , temp , obj_points);

		err_type err_code = Laser_Printer_System(img_cfg);
		CHECK_EXP_RETURN(err_code != SUCCESS, err_code);

		if (err_code != SUCCESS)
		{
			cout << "remapping failed #" << endl << endl;
		}
		else if (err_code == SUCCESS)
		{
			cout << "remapping succeed at Point : " << img_cfg.ow_points[0] << endl << endl;
		}

		static size_t n = 0;
		static double time;
		while (n++ < 10000)
		{
			Mat img_any = imread("C:\\Users\\EDZ\\Desktop\\opencv_cablication\\Image_20190613164200.jpg", IMREAD_GRAYSCALE);
			static float degree = 0;
			degree += 3;
			img_any = imrotate(img_any , degree);
		time = (double)getTickCount();
			img_config img_cfg = img_config(img_any);
			err_code = Laser_Printer_System(img_cfg);

		time = (((double)getTickCount() - time) / getTickFrequency()) * 1000;

			if ((waitKey(1) & 0xff) == 27) break;
			if (err_code != SUCCESS)
			{
			    cout << "remapping failed #" << endl << endl;
				if ((waitKey(0) & 0xff) == 27) continue;
			}
			else if(err_code == SUCCESS)
			{
				cout << "remapping succeed at Point : " << img_cfg.ow_points[0] << endl << endl;
			}
			cout << "times n =  " << n << " " << " run time of once calculation =  " << time << " ms." << endl;
		}
	}*/
/*
int main(int argc, char** argv)
{
	static size_t n = 0;
	while ( n++ < 100000 )
	{
		//auto cap = VideoCapture(0 + CAP_ARAVIS);
		static bool Calibrated = false; //相机标定 矫正只需最开始的一次，GUI可控
		static vector<Mat> vec_mat; //透视变换阵
static double time = 0;

		if (!Calibrated)
		{
			vec_mat.clear();
			for (int i = 0, try_times = 3; i < try_times; ++i)
			{
				static Mat view = imread("C:\\Users\\EDZ\\Desktop\\opencv_cablication\\chessboard.png" , IMREAD_GRAYSCALE); //入参
				//CHECK_EXP_RETURN(view.empty() , PATTERN_IMAGE_NULL);
				Size board_num(7,7);
				Size2f real_size(62.93, 65.2);

				//入参校验

				calibrate_camera(CHESSBOARD , view, real_size, board_num, vec_mat ); //相机 矫正 标定
				if (vec_mat.size() == 4)
				{
					Calibrated = true;
					break;
				}
			}
		}
		CHECK_EXP_RETURN(vec_mat.size() != 4, CALIBRATE_FAILED);

time = (double)getTickCount();

		static Mat tem = imread("C:\\Users\\EDZ\\Desktop\\opencv_cablication\\ss.png", IMREAD_GRAYSCALE);
		const double scaler = 1;
		Mat temp = tem.clone();
		CHECK_EXP_RETURN(temp.empty(), IMAGE_TEMP_INVALID);
		cv::resize(temp, temp, Size() , scaler , scaler , INTER_LINEAR);
		
		static Mat ig = imread("C:\\Users\\EDZ\\Desktop\\opencv_cablication\\Image_20190613164200.jpg", IMREAD_GRAYSCALE);;
		Mat	img = ig.clone();
		CHECK_EXP_RETURN(img.empty(), IMAGE_INVALID);
		cv::resize(img , img, Size() , scaler , scaler , INTER_LINEAR);

		vector<Point2f> src_tri;
		src_tri = matching_template_accelerated(img, temp);
		CHECK_EXP_RETURN(src_tri.size() != 3, MATCHING_TRIPPLE_FAILED);

		vector<Point2f> Obj_Points;
		Obj_Points.push_back(Point2f((rand() % 200000) / 100.0, (rand() % 200000) / 100.0));
		CHECK_EXP_RETURN(Obj_Points.empty(), OBI_EMPTY);

		vector<Point2f> Ow_Points = Obj_to_Ow(Obj_Points, src_tri, vec_mat);
		CHECK_EXP_RETURN(Ow_Points.empty(), REMMPING_FAILED);

		time = (((double)getTickCount() - time) / getTickFrequency()) * 1000;
		draw_circle(img, src_tri);
		set_lable(img , to_string((float)time) + string(" ms ") , src_tri[0]);
		cv::resize(img, img, Size(), 0.5 / scaler, 0.5 / scaler, INTER_LINEAR);
		imshow("matching img", img);
		if( (waitKey(50) & 0xff) == 27) break;

		cout << "times n =  " << n << " "<< " run time of remapping =  " << time <<" ms."<< endl;
		if(Ow_Points.size() != 0) cout << "remapping succeed at Point : " << Ow_Points[0] << endl << endl;
		else cout << "remapping failed #" << endl << endl;
	}
	waitKey(0);
	return 0;
}
*/


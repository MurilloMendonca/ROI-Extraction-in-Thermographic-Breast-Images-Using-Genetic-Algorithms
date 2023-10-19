#pragma once
#include <opencv2/opencv.hpp>

class Cardioid{
	public:
		int x;
		int y;
		int size;
	
		Cardioid(int x, int y, int size);
		Cardioid(){};
		bool isInside(int x, int y);
		void setX(int x);
		void setY(int y);
		void setSize(int size);
		int getX();
		int getY();
		int getSize();

		

		bool operator==(const Cardioid& other) const;

};
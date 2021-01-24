#include<stdio.h>
#include<iostream>
#include"Matrix.hpp"

using namespace std;

int main()
{
	
	double **disks, scalar1, eigenvalue1, eigenvalue2, S;
	string fileName;
	Matrix A, A1;
	Matrix x0;

	cout << endl << "Enter file name to accept coefficient matrix A from the file:";
	cin >> fileName;
	
	A.readInputFromFile( fileName );
	A.display();


	disks = A.gersehgorin();
	
	cout<<endl<<"disks "<<endl;
	
	for(int i = 0 ; i < A.rows ; ++i)
		cout << "D" << i+1 << "=[" << disks[i][0] << ", " << disks[i][1] << "]" << endl;
	
	//A.givenMethod();
	cout << endl << "Eigenvalues:" << endl;
	
	cout << "Tridiagonalized matrix by Given's Method is " << endl;
	A.display();
	
	A.householderMethod();
	cout << endl << "Eigenvalues:" << endl;
	
	cout << "Tridiagonalized matrix by Householder's Method is " << endl;
	A.display();
	
	return 0;
}

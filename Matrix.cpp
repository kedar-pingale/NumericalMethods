#include<iostream>
// #include<stdio.h>
#include<fstream>
#include<math.h>
#include"Matrix.hpp"
#define THRESHOLD 1e-8
using namespace std;


//Default constructor
Matrix::Matrix()
{
	this->rows = 0;
	this->cols = 0;
}


//Parameterized constructor
Matrix::Matrix(unsigned int rows, unsigned int cols)
{
	int i;
	this->rows = rows;
	this->cols = cols;
	this->matrix = new double*[rows];
	for(i=0 ; i<rows ; i++)
	{
		this->matrix[i] = new double[cols];
		for(int j=0 ; j<cols ; j++)
			matrix[i][j] = 0;
	}
}


//Copy constructor
Matrix::Matrix(Matrix &matrix1)
{
	this->rows = matrix1.rows;
	this->cols = matrix1.cols;
	this->matrix = new double*[rows];
	for(int i=0 ; i<rows ; i++)
	{
		this->matrix[i] = new double[cols];
		for(int j=0 ; j<cols ; j++)
			matrix[i][j] = matrix1.matrix[i][j];
	}
	
}


//Destructor
Matrix::~Matrix()
{
//	for(int i=0 ; i<rows ; i++)
//		delete [] this->matrix[i];
//	delete [] this->matrix;
}

	
//Member function to read input from file and write to Matrix object
void Matrix::readInputFromFile(string fileName)
{
	fstream fin;
	fin.open(fileName);
   	fin>>rows;
   	fin>>cols;
   	
   	this->matrix = new double*[rows];
   	for(int i=0 ; i<rows ; i++)
   	{
   		this->matrix[i] = new double[cols];
   		for(int j=0 ; j<cols ; j++)
   			fin>>this->matrix[i][j];
   		cout<<endl;
   	}
   	fin.close();
}


//Member function to read Martrix object and write output to a file
void Matrix::writeOutputToFile(string fileName)
{
	
	fstream fout;
	
	fout.open(fileName);
   	fout<<rows<<" ";
   	fout<<cols<<endl;
   	
   	for(int i=0 ; i<rows ; i++)
   	{
   		for(int j=0 ; j<cols ; j++)
   			fout<<this->matrix[i][j]<<" ";
   		fout<<"\n";
   	}
   	fout.close();
}


//Member function to write Matrix values on console i.e. to display Matrix object 
void Matrix::display()
{
	int i, j;
	for(i=0 ; i<this->rows ; i++)
	{
		for(j=0 ; j<this->cols ; j++)
		{
			cout<<this->matrix[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl<<"Dim:"<<rows<<"x"<<cols<<endl;
}


//Checks if given matrix is an identity matrix
bool Matrix::isIdentity()
{
	for(int i=0 ; i<rows ; i++)
	{
		for(int j=0 ; j<cols ; j++)
		{
			if(i == j && matrix[i][j] != 1)
				return false;
			else if(i != j && matrix[i][j] != 0)
				return false;
		}
	}
	return true;
}


ostream& operator<<(ostream &out, Matrix object)
{
	out<<object.rows<<" "<<object.rows<<endl;
	for(int i=0 ; i<object.rows ; i++)
	{
		for(int j=0 ; j<object.cols ; j++)
			out<<object[i][j]<<" ";
		out<<endl;
	}
	return out;
}


Matrix Matrix::operator+(Matrix matrix1)
{
	Matrix tempMatrix(this->rows, this->cols);
	for(int i=0 ; i<matrix1.rows ; i++)
	{
		for(int j=0 ; j<matrix1.cols ; j++)
		{
			tempMatrix.matrix[i][j] = this->matrix[i][j] + matrix1.matrix[i][j];
		}
	}
	return tempMatrix;
}


Matrix Matrix::operator-(Matrix matrix1)
{
	Matrix tempMatrix(this->rows, this->cols);
	for(int i=0 ; i<matrix1.rows ; i++)
	{
		for(int j=0 ; j<matrix1.cols ; j++)
		{
			tempMatrix.matrix[i][j] = this->matrix[i][j] - matrix1.matrix[i][j];
		}
	}
	return tempMatrix;
}


//scalar multiplication
Matrix Matrix::operator*(double scalar1)
{
	Matrix tempMatrix(this->rows, this->cols);
	for(int i=0 ; i<this->rows ; i++)
	{
		for(int j=0 ; j<this->cols ; j++)
		{
			tempMatrix.matrix[i][j] = this->matrix[i][j] * scalar1;
		}
	}	
	return tempMatrix;
}


//matrix matrix multiplication
Matrix Matrix::operator*(Matrix matrix1)
{
	Matrix tempMatrix(this->rows, matrix1.cols);
	for(int i=0 ; i<this->rows ; i++)
	{
		for(int j=0 ; j<matrix1.cols ; j++)
		{
			tempMatrix.matrix[i][j] = 0;
			for(int k=0 ; k<this->cols ; k++)
			{
				tempMatrix.matrix[i][j] += (this->matrix[i][k] * matrix1.matrix[k][j]);
			}	
		}
	}
	return tempMatrix;
}


Matrix Matrix::operator/(double scalar1)
{
	Matrix tempMatrix(this->rows, this->cols);
	for(int i=0 ; i<this->rows ; i++)
	{
		for(int j=0 ; j<this->cols ; j++)
		{
			tempMatrix.matrix[i][j] = this->matrix[i][j] / scalar1;
		}
	}	
	return tempMatrix;
}


bool Matrix::operator==(const Matrix &matrix1)
{
	if(rows != matrix1.rows || cols != matrix1.cols)
		return false;
	for(int i=0 ; i<rows ; ++i)
		for(int j=0 ; j<cols ; ++j)
			if(abs(matrix[i][j] - matrix1.matrix[i][j]) > THRESHOLD)
				return false;
	return true;
}


double*& Matrix::operator[](unsigned int i)
{
	return matrix[i];
}


Matrix::operator double()
{
	double sum = 0;
	for(int i=0 ; i<rows ; i++)
		for(int j=0 ; j<rows ; j++)
			sum += matrix[i][j];
	return sum;
}


bool Matrix::isSymmetric()
{
	int limit = rows-1;
	for(int i=0 ; i<limit ; i++)
	{
		for(int j=i+1 ; j<cols ; j++)
		{
			if(matrix[i][j] != matrix[j][i])
				return false;
		}
	}
	return true;
}


bool Matrix::isNull()
{
	for(int i=0 ; i<rows ; i++)
	{
		for(int j=0 ; j<cols ; j++)
		{
			if(matrix[i][j] != 0)
				return false;
		}
	}
	return true;
}


bool Matrix::isDiagonal()
{
	for(int i=0 ; i<rows ; i++)
	{
		for(int j=0 ; j<cols ; j++)
		{
			if(i != j && matrix[i][j] != 0)
				return false;
		}	
	}
	return true;
}


bool Matrix::isDiagonallyDominant()
{
	double sum;
	for(int i=0 ; i<rows ; i++)
	{
		sum = 0;
		for(int j=0 ; j<cols ; j++)
		{
			if(i != j)
			sum += abs(matrix[i][j]);
		}
		if(sum > abs(matrix[i][i]))
			return false;
	}
	return true;
}


Matrix Matrix::GaussianElimination(Matrix &b)
{
	double tempElement;
	
	if (matrix[0][0] != 1) {				//for making leading coefficient in 1st row 1(if it's not 1)
		tempElement = matrix[0][0];
		for(int j = 0 ; j < cols ; j++)
			matrix[0][j] /= tempElement;
		b.matrix[0][0] /= tempElement;
	}
	
	for(int i=0 ; i<rows-1 ; )	//for making augmented matrix [A|b] upper triangular
	{
		for(int j=i+1 ; j<rows ; j++)	//Doing operations on all rows for making elements below leading coefficient 0
		{
			tempElement = matrix[j][i];
			
			if(tempElement != 0)
			{
				matrix[j][i] = 0;
				
				for(int k=i+1 ; k<cols ; k++)	//making changes to a row of A as we've made element below leading coeff. 0
					matrix[j][k] -= tempElement * matrix[i][k];
				b.matrix[j][0] -= tempElement * b.matrix[i][0];   //for updating b matrix row as per row 											    operation			
			}
		}
		
		i++;	//incrementing i for making next row's leading coefficient 1(if it's not 1)
		if((tempElement = matrix[i][i]) != 1 && tempElement != 0)
		{
			matrix[i][i] = 1;
			for(int j=i+1 ; j<cols ; j++)	//dividing each element of row with proper scalar so as to make leading coeff. 1
				matrix[i][j] /= tempElement;
			b.matrix[i][0] /= tempElement;
		}
	}

	for(int i=rows-1 ; i>0 ; i--)	//for making augmented matrix [A|b] lower triangular
	{	
		for(int j=i-1 ; j>=0 ; j--)	//Doing operations on some rows for making elements above leading coefficient 0
		{		
			tempElement = matrix[j][i];
			
			if(tempElement != 0)
			{
				matrix[j][i] = 0;		
				for(int k=i+1 ; k<cols ; k++)	//making changes to a row of A as we've made element above leading coeff. 0
					matrix[j][k] -= tempElement * matrix[i][k];
				b.matrix[j][0] -= tempElement * b.matrix[i][0];   //for updating b matrix row as per row operation
			}
		}		
	}
	return (*this);
}


bool Matrix::isSquareMatrix()
{
	if(this->rows != this->cols)
		return false;
	return true;
}


Matrix Matrix::transpose()
{
	cout<<"display in transp:"<<endl;
	this->display();
	Matrix tempMatrix(cols, rows);
	double temp;
	tempMatrix.display();
	for(int i=0 ; i<rows ; i++)
	{
		for(int j=0 ; j<1 ; j++)
		{
			
			tempMatrix[j][i] = this->matrix[i][j];
		}
	}

	tempMatrix.display();
	return tempMatrix;
}


//trace of the matrix is sum of it's diagonal elements
double Matrix::trace()
{
	double traceValue = 0;
	for(int i=0 ; i<rows ; i++)
		traceValue += matrix[i][i];
	return traceValue;
}


//when M1*transpose(M1) and transpose(M1)*M1 = I, then matrix M1 is orthogonal
bool Matrix::isOrthogonal()
{
	Matrix temp(*this);
	Matrix temp2;
	temp2 = this->transpose();
	
	Matrix m1, m2; 
	m1 = temp*temp2;
	m2 = temp2*temp;
	if(m1.isIdentity() && m2.isIdentity())
		return true;
	return false;
}


//iterative method to find solution to system of linear equations by using values obtained from previous step. Condition:matrix A must be diagonally dominant
Matrix Matrix::gaussJacobi(Matrix b)
{
	float difference;
	int iterations = 1;
	Matrix x(b.rows, b.cols);
	Matrix xtemp(x);
	
	for(int i=0 ; i<rows ; ++i)
		x.matrix[i][0] = b[i][0] / matrix[i][i];
	
	do
	{
		iterations++;
		
		difference = 0;
		for(int i=0 ; i<rows ; ++i)
		{
			xtemp.matrix[i][0] = b.matrix[i][0];
			for(int j=0 ; j<cols ; j++)
			{
				if(i!=j)
					xtemp.matrix[i][0] = xtemp.matrix[i][0] - this->matrix[i][j] * x.matrix[j][0];
			}
			xtemp.matrix[i][0] /= this->matrix[i][i];
			
		}
		
		for(int i=0 ; i<rows ; ++i)
		{
			if(abs(xtemp.matrix[i][0] - x.matrix[i][0]) > difference)
				difference = abs(xtemp.matrix[i][0] - x.matrix[i][0]);
			x.matrix[i][0] = xtemp.matrix[i][0];
		}
		
	}while(difference > THRESHOLD);

	return x;
}


//iterative method to find solution to system of linear equations by using latest updated values. Condition:matrix A must be diagonally dominant
Matrix Matrix::gaussSiedel(Matrix b)
{
	float difference;
	int iterations = 0;
	Matrix x(b.rows, b.cols);
	Matrix xtemp(x);
	
	do
	{
		iterations++;
		difference = 0;
		for(int i=0 ; i<rows ; ++i)
		{
			xtemp.matrix[i][0] = b.matrix[i][0];
			for(int j=0 ; j<cols ; j++)
			{
				if(i!=j)
					xtemp.matrix[i][0] = xtemp.matrix[i][0] - this->matrix[i][j] * x.matrix[j][0];
			
			}
			xtemp.matrix[i][0] /= this->matrix[i][i];
			if(abs(xtemp.matrix[i][0] - x.matrix[i][0]) > difference)
				difference = abs(xtemp.matrix[i][0] - x.matrix[i][0]);
			x.matrix[i][0] = xtemp.matrix[i][0];
		}
	}while(difference > THRESHOLD);
	
	//cout<<endl<<"For gauss siedel, iterations required are "<<iterations;
	
	return x;
}


Matrix Matrix::inverseByGaussianElimination()
{
	double tempElement;
	Matrix temp;
	temp = Matrix::getIdentityMatrix(this->rows);
	
	if (matrix[0][0] != 1) {				//for making leading coefficient in 1st row 1(if it's not 1)
		tempElement = matrix[0][0];
		for(int j = 0 ; j < cols ; j++)
		{
			matrix[0][j] /= tempElement;
		}
		temp[0][0]/=tempElement;
	}
	
	for(int i=0 ; i<rows-1 ; )	//for making augmented matrix [A|b] upper triangular
	{
		for(int j=i+1 ; j<rows ; j++)	//Doing operations on all rows for making elements below leading coefficient 0
		{
			tempElement = matrix[j][i];
			if(tempElement != 0)
			{
				matrix[j][i] = 0;
				
				for(int k=i+1 ; k<cols ; k++)	//making changes to a row of A as we've made element below leading coeff. 0
					matrix[j][k] -= tempElement * matrix[i][k];
				for(int k=0 ; k<i+1 ; k++)	//for making transformations on rows of identity matrix
					temp[j][k] -= tempElement * temp[i][k];	
			}
		}
		
		i++;	//incrementing i for making next row's leading coefficient 1(if it's not 1)
		if((tempElement = matrix[i][i]) != 1 && tempElement != 0)
		{
			matrix[i][i] = 1;
			for(int j=i+1 ; j<cols ; j++)	//dividing each element of row with proper scalar so as to make leading coeff. 1
				matrix[i][j] /= tempElement;
			for(int j=0 ; j<i+1 ; j++)	//dividing each element of row with proper scalar so as to make leading coeff. 1
				temp[i][j] /= tempElement;
		}
	}

	for(int i=rows-1 ; i>0 ; i--)	//for making augmented matrix [A|b] lower triangular
	{	
		for(int j=i-1 ; j>=0 ; j--)	//Doing operations on some rows for making elements above leading coefficient 0
		{		
			tempElement = matrix[j][i];
			
			if(tempElement != 0)
			{
				matrix[j][i] = 0;		
				//for(int k=i+1 ; k<cols ; k++)	making changes to a row of A as we've made element above leading coeff. 0
				//	matrix[j][k] -= tempElement * matrix[i][k];
				for(int k=0 ; k<cols ; k++)
					temp[j][k] -= tempElement * temp[i][k];
			}
		}		
	}
	return temp;
}


Matrix Matrix::getIdentityMatrix(int dim)
{
	Matrix temp(dim, dim);
	for(int i=0 ; i<temp.rows ; i++)
		temp[i][i] = 1;
	return temp;
}


//To find eigenvalue with the biggest magnitude
double Matrix::powerMethod(Matrix x)
{
	int iterations=0;
	double eMax=0, previousEMax;
	Matrix xtemp(x.rows, x.cols);
	xtemp = (*this) * x;
	
	do
	{	
		previousEMax = eMax;
		eMax = xtemp[0][0];
		for(int i=1 ; i<xtemp.rows ; ++i)
			if(abs(eMax) < abs(xtemp[i][0]))
				eMax = xtemp[i][0];
		for(int i=0 ; i<xtemp.rows ; ++i)
			xtemp[i][0] = xtemp[i][0] / eMax;
		xtemp = (*this) * xtemp;
		++iterations;
		
	}while(iterations<2 || abs(previousEMax - eMax) > THRESHOLD);
	return eMax;
}


double Matrix::shiftedInversePowerMethod(Matrix x, double S)
{
	Matrix mat1, inverse;
	mat1 = Matrix::getIdentityMatrix(this->rows);
	mat1 = mat1 * S;
	mat1 = (*this) - mat1;
	inverse = mat1.inverseByGaussianElimination();
	double eMax;
	eMax = inverse.powerMethod(x);
	
	return S + 1/eMax;
}


double** Matrix::gersehgorin()
{
	
	double sum, **disks;
	
	disks = new double*[rows];
	
	for(int i=0 ; i<rows ; i++)
	{
		disks[i] = new double[cols];
		sum = 0;
		
		for(int j=0 ; j<cols ; j++)
			if(i != j)
				sum += abs(this->matrix[i][j]);
		disks[i][0] = this->matrix[i][i] - sum;
		disks[i][1] = this->matrix[i][i] + sum;
	}
	return disks;
}


Matrix Matrix::jacobiForEigenvalues()
{
	unsigned int iterations = 0, maxi = 0, maxj = 1;
	double tan2Value, cosValue, sinValue;
	Matrix temp(*this);
	
	do
	{
		iterations++;
		for(int i=0 ; i<rows ; ++i)
		{
			for(int j=0 ; j<cols ; ++j)
			{
				if(i != j && abs(this->matrix[i][j]) > this->matrix[maxi][maxj])
				{
					maxi = i;
					maxj = j;
				}
			}
		}
		
		if(abs(this->matrix[maxi][maxj]) < THRESHOLD || iterations > 100)
			break;
		
		tan2Value = (2 * this->matrix[maxi][maxj])/(this->matrix[maxi][maxi]-this->matrix[maxj][maxj]);
		
		cosValue = sqrt( (1/2.0) * ( 1 + ( 1 / ( 1 + (tan2Value * tan2Value) ) ) ) );
		sinValue = sqrt( (1/2.0) * ( 1 - ( 1 / ( 1 + (tan2Value * tan2Value) ) ) ) );


		temp.matrix[maxi][maxi] = this->matrix[maxi][maxi] * (cosValue * cosValue) + 2 * this->matrix[maxi][maxj] * sinValue * cosValue + this->matrix[maxj][maxj] * (sinValue * sinValue);
		temp.matrix[maxj][maxj] = this->matrix[maxi][maxi] * (sinValue * sinValue) - 2 * this->matrix[maxi][maxj] * sinValue * cosValue + this->matrix[maxj][maxj] * (cosValue * cosValue);
		temp.matrix[maxi][maxj] = temp.matrix[maxj][maxi] = (this->matrix[maxj][maxj] - this->matrix[maxi][maxi])*(cosValue*sinValue)+this->matrix[maxi][maxj] * ((cosValue * cosValue) - (sinValue * sinValue));
		
		for(int i=0 ; i<rows ; i++)
		{
			if(i != maxi && i != maxj)
			{
				temp.matrix[maxi][i] = this->matrix[maxi][i] * cosValue + this->matrix[maxj][i] * sinValue;
				temp.matrix[maxj][i] = -this->matrix[maxi][i] * sinValue + this->matrix[maxj][i] * cosValue;
				
				temp.matrix[i][maxi] = this->matrix[i][maxi] * cosValue + this->matrix[i][maxj] * sinValue;
				temp.matrix[i][maxj] = -this->matrix[i][maxi] * sinValue + this->matrix[i][maxj] * cosValue;
			}
		}
	
		for(int i=0 ; i<rows ; ++i)
		{
			for(int j=0 ; j<cols ; ++j)
			{
				this->matrix[i][j] = temp.matrix[i][j];
				this->matrix[j][i] = temp.matrix[j][i];
			}
		}	
	}while(true);	
	return *this;
}


void Matrix::givenMethod()
{
	float tanValue, sinValue, cosValue;
	int p, q;
	Matrix temp(*this);
	

	for(int k=0 ; k<rows-2 ; ++k)
	{
		for(int l=k+2 ; l<cols ; ++l)
		{
			p = k;
			q = l;
			tanValue = this->matrix[p][q]/this->matrix[p][p+1];
	
			cosValue = 1/sqrt(1 + tanValue*tanValue);
			sinValue = sqrt(1 - cosValue*cosValue);
			
			p++;
			
			for(int i=0 ; i<rows ; ++i)
			{
				
				
				if(i != p && i != q)
				{
					temp.matrix[p][p] = this->matrix[p][p] * (cosValue * cosValue) + 2 * this->matrix[p][q] * sinValue * cosValue + this->matrix[q][q] * (sinValue * sinValue);
					temp.matrix[q][q] = this->matrix[p][p] * (sinValue * sinValue) - 2 * this->matrix[p][q] * sinValue * cosValue + this->matrix[q][q] * (cosValue * cosValue);
					temp.matrix[p][q] = temp.matrix[q][p] = (this->matrix[q][q] - this->matrix[p][p])*(cosValue*sinValue)+this->matrix[p][q] * ((cosValue * cosValue) - (sinValue * sinValue));
					
					for(int i=0 ; i<rows ; i++)
					{
						if(i != p && i != q)
						{
							temp.matrix[p][i] = this->matrix[p][i] * cosValue + this->matrix[q][i] * sinValue;
							temp.matrix[q][i] = -this->matrix[p][i] * sinValue + this->matrix[q][i] * cosValue;
							
							temp.matrix[i][p] = this->matrix[i][p] * cosValue + this->matrix[i][q] * sinValue;
							temp.matrix[i][q] = -this->matrix[i][p] * sinValue + this->matrix[i][q] * cosValue;
						}
					}
				
					for(int i=0 ; i<rows ; ++i)
					{
						for(int j=0 ; j<cols ; ++j)
						{
							this->matrix[i][j] = temp.matrix[i][j];
							this->matrix[j][i] = temp.matrix[j][i];
						}
					}
				}
			}
		}
	}
}


int Matrix::sign(double number)
{
	if(number < 0)
		return -1;
	else if(number > 0)
		return 1;
	return 0;
}


void Matrix::householderMethod()
{
	float S, sum;
	
	Matrix temp(this->rows, 1), P(rows, cols);
		
	
	for(int i=0 ; i<rows-2 ; ++i)
	{
		sum = 0;
		for(int j=i+1 ; j<cols ; ++j)
		{
			sum += (this->matrix[i][j] * this->matrix[i][j]);
		}
		
		S = sqrt(sum);
		
		for(int j=0 ; j<=i ; ++j)
			temp[j][0] = 0;
		for(int j=i+1 ; j<cols ; ++j)
		{
			if(j == i+1)
			{
				temp[j][0] = sqrt((1/2.0) * (1 + sign(this->matrix[i][i+1]) * this->matrix[i][i+1] / S));
			}
			
			else
				temp[j][0] = (sign(this->matrix[i][i+1]) * this->matrix[i][j]) / (2 * S * temp.matrix[i+1][0]);
			cout<<"tempval="<<temp.matrix[j][0]<<endl;
		}
		cout<<"X:"<<endl;
		temp.display();
		for(int j=0 ; j<rows ; j++)
		{
			cout<<temp.matrix[j][0]<<endl;
		}
		
		Matrix transp, t1, t2; 
		transp = temp.transpose();
		
		t1 = (temp*transp);
		t2 = t1*2.0;
		P = ((getIdentityMatrix(this->rows )) - t2);
		
		*this = P * (*this) * P;
		this->display();
	}
}

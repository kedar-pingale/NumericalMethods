using namespace std;

class Matrix
{
	public: unsigned int rows;
	unsigned int cols;
	double **matrix;

	Matrix();
	Matrix(unsigned int rows, unsigned int cols);
	Matrix(Matrix &matrix1);
	~Matrix();
	
	void readInputFromFile(string fileName);
	void writeOutputToFile(string fileName);
	
	void display();
	
	Matrix operator+(Matrix matrix1);
	Matrix operator-(Matrix matrix1);
	Matrix operator*(double scalar1);
	Matrix operator*(Matrix matrix1);
	Matrix operator/(double scalar1);
	bool operator==(const Matrix &matrix1);
	operator double();
	
	double*& operator[](unsigned int i);
	
	bool isIdentity();
	bool isSymmetric(); 
	bool isNull();
	bool isDiagonal();
	bool isDiagonallyDominant();

	bool isSquareMatrix();
	Matrix transpose();
	double trace();
	bool isOrthogonal();
	
	int rowForBasicPivoting(int row);
	void interchangeRows(int row1, int row2);
	Matrix GaussianElimination(Matrix &b);
	Matrix gaussJacobi(Matrix b);
	Matrix gaussSiedel(Matrix b);

	static Matrix getIdentityMatrix(int);
	Matrix inverseByGaussianElimination();
	double powerMethod(Matrix x);
	double shiftedInversePowerMethod(Matrix x, double S);
		
	friend ostream& operator<<(ostream &out, Matrix object);
	
	double** gersehgorin();
	
	Matrix jacobiForEigenvalues();
	void givenMethod();
	void householderMethod();
	int sign(double number);
	
	//void reduce(unsigned int p, unsigned int q);
};

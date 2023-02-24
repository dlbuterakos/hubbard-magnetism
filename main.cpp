#include <cstdio>
#include <math.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <sstream>
#include <fstream>
using namespace std;

/*
 Represents the state of a quantum dot (Hubbard model site).
 EMPTY: No electrons are in the quantum dot. Represents state: |0>
 UP: One electron with spin up is in the quantum dot. Represents state: c^dagger_up |0>
 DOWN: One electron with spin down is in the quantum dot. Represents state: c^dagger_down |0>
 Two electrons do not occupy the same dot due to the large Coulombic repulsion U.
 If the system is more than half-filled, instead model a system less than half-filled and use particle-hole transformation.
 */
enum class DotState { 
	EMPTY,  // No electrons are in the quantum dot. Represents state: |0>
	UP,     // One electron with spin up is in the quantum dot. Represents state: c^dagger_up |0>
	DOWN    // One electron with spin down is in the quantum dot. Represents state: c^dagger_down |0>
};

/*
Semi-sparse Matrix meant for storing NxN matrices where N is large, that have on the order of N nonzero entries.
The SparseMatrix is stored as a size-N vector, where the i-th element of the vector
is a conventional sparse array representing the i-th row of the matrix.
*/
class SparseMatrix {
private:

	/*
	(index, value) pair for the 1D sparse array comprising a row of the SparseMatrix.
	*/
	struct Entry {
		int index;
		double value;
		Entry(int ind, double val) { index = ind; value = val; }
	};

	int size_; // This is the value N of the NxN matrix
	/*
	data is a size N-vector, where data[i] is a conventional 1D sparse array which represents the i-th row of the matrix.
	data[i] is a vector whose size is equal to the number of nonzero elements in row i of the matrix.
	data[i] contains elements of the form Entry(j, value),
	where value is a double representing the element of the matrix at row i and column j.
	If there is no entry for a particular value of i and j, then the corresponding value of the matrix is assumed to be 0.
	*/
	vector<vector<Entry>> data;

public:

	/*
	Initializes a 0x0 SparseMatrix.
	The SparseMatrix must be resized via the method clear(int) before it can be meaningfully used.
	*/
	SparseMatrix() {
		this->size_ = 0;
	}

	/*
	Initializes a size x size SparseMatrix. Initially all entries of the SparseMatrix are 0.
	*/
	SparseMatrix(int size) :
		data(size) {
		this->size_ = size;
	}

	/*
	Clears all data from the SparseMatrix and resises it to the the specified size.
	After calling this method, the SparseMatrix will be a size x size matrix with all entries equal to 0.
	*/
	void clear(int size) {
		this->size_ = size;
		data.resize(size);
		for (vector<Entry>& v : data) { v.clear(); }
	}

	/*
	Adds an entry value to the SparseMatrix at row i and column j.

	WARNING: add(i, j, value) must NOT be called more than once for a given pair (i, j)
	unless clear(int) has been called since then.
	This method will not check to ensure add(i, j, value) is not called multiple times for the same pair (i, j).
	*/
	void add(int i, int j, double value) {
		data[i].emplace_back(j, value);
	}

	/*
	Performs the matrix operation out = A v, where A is this SparseMatrix and v is a vector of the same size as A.

	WARNING: v MUST be the same size as this SparseMatrix.
	This method will not check to ensure that v is the proper size.
	*/
	vector<double> times(const vector<double>& v) const {
		vector<double> out(v.size());
		double d;
		for (int i = 0; i < static_cast<int>(v.size()); i++) {
			d = 0.0;
			for (const Entry& e : data[i]) {
				d += v[e.index] * e.value;
			}
			out[i] = d;
		}
		return move(out);
	}

	/*
	Returns the size N of this NxN SparseMatrix.
	*/
	int size() const { return size_; }

	/*
	Multiplies every value in this SparseMatrix by a double d.
	*/
	void multiplyby(double d) {
		for (vector<Entry>& v : data) {
			for (Entry& e : v) {
				e.value *= d;
			}
		}
	}

	/*
	Converts this SparseMatrix to a NxN matrix, where all elements including zeros are explicitly listed.
	The output is given as a vector<vector<double>>, where out and out[i] have size given by size().
	WARNING: Do not call this method on a large SparseMatrix
	*/
	vector<vector<double>> normalform() const {
		vector<vector<double>> out(size_);
		for (int i = 0; i < size_; i++) {
			out[i] = vector<double>(size_, 0.0);
			for (const Entry& e : data[i]) {
				out[i][e.index] = e.value;
			}
		}
		return move(out);
	}
	
};

/*
Represents a plaquette of quantum dots.
Contains the name of the Plaquette, a bool indicating whether it is bipartite,
and a SparseMatrix which contains the Hamiltonian tunneling elements between pairs of dots.

WARNING: Plaquette does not check whether the plaquette is in fact bipartite.
*/
struct Plaquette {
public:

	string name;
	bool isBipartite;
	SparseMatrix h;

	/*
	Initializes the Plaquette, specifying all fields directly.
	*/
	Plaquette(const string& name, bool isBipartite, const SparseMatrix& h) {
		this->name = name;
		this->isBipartite = isBipartite;
		this->h = h;
	}

	/*
	Initializes the Plaquette. Rather than specify h directly,
	instead constructs h from a set of edges assuming equal weighting of all edges.
	Each nonzero element of h will be equal to value.
	edges is a vector with size equal to the number of dots in the plaquette (this is the size of h as well).
	Assume the quantum dots are numbered starting with 1 (for easy compatibility with Mathematica).
	Thus edges[i] represents dot i+1
	edges[i] is a vector of integers with size equal to the number of dots adjacent to dot i+1.
	Each element of edges[i] is the number of a dot which is connected to dot i+1.
	*/
	Plaquette(const string& name, bool isBipartite, double value, const vector<vector<int>>& edges) {
		this->name = name;
		this->isBipartite = isBipartite;
		int size = edges.size();
		this->h = SparseMatrix(size);
		for (int i = 0; i < size; i++) {
			for (int j : edges[i]) {
				h.add(i, j - 1, value);
			}
		}
	}

	/*
	Returns the number of dots in the plaquette.
	*/
	int size() { return h.size(); }

};

class Main {
private:

	// Binomial coefficients are calculated once and then cached 
	vector<vector<int>> binomialcache;

	// Random number generator for generating initial vectors.
	// Generating each component with a normal distribution ensures that the direction of the vector is uniformly random
	random_device rd{};
	mt19937 rgen{ rd() };
	normal_distribution<> normdist{ 0.0, 1.0 };

	// List of some Plaquettes. Additional plaquettes can be appended to this list.
	// When calling run(vector<int>), a vector<int> is supplied to specify which plaquettes to calculate.
	vector<Plaquette> plaquettes{
		Plaquette("9 dots: (2,3)-Hamming graph", false, 1.0, vector<vector<int>> { // index = 0
			{2,3,4,7}, {1,3,5,8}, {1,2,6,9}, {1,5,6,7}, {2,4,6,8}, {3,4,5,9}, {1,4,8,9}, {2,5,7,9}, {3,6,7,8}}),
		Plaquette("10 dots: Petersen graph (pentagons with PBC)", false, 1.0, vector<vector<int>> {
			{2,5,9}, {1,3,7}, {2,4,10}, {3,5,8}, {1,4,6}, {5,7,10}, {2,6,8}, {4,7,9}, {1,8,10}, {3,6,9}}),
		Plaquette("11 dots: Kagome lattice", false, 1.0, vector<vector<int>> {
			{2,6,7}, {1,3}, {2,4}, {3,5}, {4,6,11}, {1,5,7,11}, {1,6,8}, {7,9}, {8,10}, {9,11}, {5,6,10}}),
		Plaquette("12 dots: 3x4 square lattice", true, 1.0, vector<vector<int>> {
			{2,4}, {1,3,5}, {2,6}, {1,5,7}, {2,4,6,8}, {3,5,9}, {4,8,10}, {5,7,9,11}, {6,8,12}, {7,11}, {8,10,12}, {9,11}}),
		Plaquette("12 dots: triangle lattice", false, 1.0, vector<vector<int>> {
			{2,4,5}, {1,3,5,6}, {2,6,7}, {1,5,8}, {1,2,4,6,8,9}, {2,3,5,7,9,10}, {3,6,10}, {4,5,9,11}, {5,6,8,10,11,12}, {6,7,9,12}, {8,9,12}, {9,10,11}}),
		Plaquette("12 dots: 4 adjacent pentagons", false, 1.0, vector<vector<int>> { // index = 5
			{2,8}, {1,3,6}, {2,4}, {3,5}, {4,6,12}, {2,5,7}, {6,8,11}, {1,7,9}, {8,10}, {9,11}, {7,10,12}, {5,11}}),
		Plaquette("12 dots: icosahedron", false, 1.0, vector<vector<int>> {
			{2,3,4,5,6}, {1,3,4,7,8}, {1,2,5,7,9}, {1,2,6,8,10}, {1,3,6,9,11}, {1,4,5,10,11}, {2,3,8,9,12}, {2,4,7,10,12}, {3,5,7,11,12}, {4,6,8,11,12}, {5,6,9,10,12}, {7,8,9,10,11}}),
		Plaquette("12 dots: cuboctahedron", false, 1.0, vector<vector<int>> {
			{2,4,5,6}, {1,3,6,7}, {2,4,7,8}, {1,3,5,8}, {1,4,9,10}, {1,2,10,11}, {2,3,11,12}, {3,4,9,12}, {5,8,10,12}, {5,6,9,11}, {6,7,10,12}, {7,8,9,11}}),
		Plaquette("12 dots: 2 adjacent septagons", false, 1.0, vector<vector<int>> {
			{2,7}, {1,3}, {2,4}, {3,5}, {4,6}, {5,7,12}, {1,6,8}, {7,9}, {8,10}, {9,11}, {10,12}, {6,11}}),
		Plaquette("12 dots: truncated tetrahedron", false, 1.0, vector<vector<int>> {
			{2,3,12}, {1,3,8}, {1,2,4}, {3,5,6}, {4,6,11}, {4,5,7}, {6,8,9}, {2,7,9}, {7,8,10}, {9,11,12}, {5,10,12}, {1,10,11}}),
		Plaquette("13 dots: FCC lattice (cuboctahedron with center)", false, 1.0, vector<vector<int>> { // index = 10
			{2,4,5,6,13}, {1,3,6,7,13}, {2,4,7,8,13}, {1,3,5,8,13}, {1,4,9,10,13}, {1,2,10,11,13}, {2,3,11,12,13}, {3,4,9,12,13}, {5,8,10,12,13}, {5,6,9,11,13}, {6,7,10,12,13}, {7,8,9,11,13}, {1,2,3,4,5,6,7,8,9,10,11,12}}),
		Plaquette("13 dots: 3 adjacent hexagons", true, 1.0, vector<vector<int>> {
			{2,6,10}, {1,3,13}, {2,4}, {3,5}, {4,6}, {1,5,7}, {6,8}, {7,9}, {8,10}, {1,9,11}, {10,12}, {11,13}, {2,12}}),
		Plaquette("13 dots: 3 septagons sharing 2 edges", false, 1.0, vector<vector<int>> {
			{2,7,11}, {1,3}, {2,4,13}, {3,5}, {4,6}, {5,7,8}, {1,6}, {6,9}, {8,10}, {9,11,12}, {1,10}, {10,13}, {3,12}}),
		Plaquette("13 dots: Paley-13 graph (triangle lattice with PBC)", false, 1.0, vector<vector<int>> {
			{2,4,5,10,11,13}, {1,3,5,6,11,12}, {2,4,6,7,12,13}, {1,3,5,7,8,13}, {1,2,4,6,8,9}, {2,3,5,7,9,10}, {3,4,6,8,10,11}, {4,5,7,9,11,12}, {5,6,8,10,12,13}, {1,6,7,9,11,13}, {1,2,7,8,10,12}, {2,3,8,9,11,13}, {1,3,4,9,10,12}}),
		Plaquette("14 dots: diamond lattice", true, 1.0, vector<vector<int>> {
			{2,5}, {1,3,8}, {2,4,7}, {3,10}, {1,6,11}, {5,7}, {3,6,13}, {2,9,12}, {8,10}, {4,9,14}, {5,12}, {8,11,13}, {7,12,14}, {10,13}}),
		Plaquette("14 dots: FCC lattice (cube corners and faces)", false, 1.0, vector<vector<int>> { // index = 15
			{2,3,4,5,7,8,9,10}, {1,3,5,6,7,10,11,14}, {1,2,4,6,7,8,11,12}, {1,3,5,6,8,9,12,13}, {1,2,4,6,9,10,13,14}, {2,3,4,5,11,12,13,14},
			{1,2,3}, {1,3,4}, {1,4,5}, {1,2,5}, {2,3,6}, {3,4,6}, {4,5,6}, {2,5,6}}),
		Plaquette("14 dots: Heawood graph (hexagons with PBC)", true, 1.0, vector<vector<int>> {
			{2,6,14}, {1,3,11}, {2,4,8}, {3,5,13}, {4,6,10}, {1,5,7}, {6,8,12}, {3,7,9}, {8,10,14}, {5,9,11}, {2,10,12}, {7,11,13}, {4,12,14}, {1,9,13}}),
		Plaquette("15 dots: 6 adjacent pentagons", false, 1.0, vector<vector<int>> {
			{2,5,14}, {1,3,12}, {2,4,10}, {3,5,8}, {1,4,6}, {5,7,15}, {6,8}, {4,7,9}, {8,10}, {3,9,11}, {10,12}, {2,11,13}, {12,14}, {1,13,15}, {6,14}}),
		Plaquette("15 dots: Kagome lattice", false, 1.0, vector<vector<int>> {
			{2,6,11,15}, {1,3,15}, {2,4}, {3,5}, {4,6,7}, {1,5,7,11}, {5,6,8}, {7,9}, {8,10}, {9,11,12}, {1,6,10,12}, {10,11,13}, {12,14}, {13,15}, {1,2,14}}),
		Plaquette("15 dots: BCC lattice(cube and adjacent centers)", true, 1.0, vector<vector<int>> {
			{7,8,9,10}, {7,10,11,14}, {7,8,11,12}, {8,9,12,13}, {9,10,13,14}, {11,12,13,14}, {1,2,3,15}, {1,3,4,15}, {1,4,5,15}, {1,2,5,15}, {2,3,6,15}, {3,4,6,15}, {4,5,6,15}, {2,5,6,15}, {7,8,9,10,11,12,13,14}}),
		Plaquette("15 dots: 4 septagons sharing 2 edges", false, 1.0, vector<vector<int>> { // index = 20
			{2,7}, {1,3,14}, {2,4}, {3,5}, {4,6,11}, {5,7}, {1,6,8}, {7,9}, {8,10,15}, {9,11}, {5,10,12}, {11,13}, {12,14}, {2,13,15}, {9,14}}),
		Plaquette("16 dots: 4x4 square lattice", true, 1.0, vector<vector<int>> {
			{2,5}, {1,3,6}, {2,4,7}, {3,8}, {1,6,9}, {2,5,7,10}, {3,6,8,11}, {4,7,12}, {5,10,13}, {6,9,11,14}, {7,10,12,15}, {8,11,16}, {9,14}, {10,13,15}, {11,14,16}, {12,15}}),
		Plaquette("16 dots: hypercube (4x4 square lattice with PBC)", true, 1.0, vector<vector<int>> {
			{2,4,5,13}, {1,3,6,14}, {2,4,7,15}, {1,3,8,16}, {1,6,8,9}, {2,5,7,10}, {3,6,8,11}, {4,5,7,12}, {5,10,12,13}, {6,9,11,14}, {7,10,12,15}, {8,9,11,16}, {1,9,14,16}, {2,10,13,15}, {3,11,14,16}, {4,12,13,15}}),
		Plaquette("16 dots: Mobius-Kantor graph (hexagons with PBC)", true, 1.0, vector<vector<int>> {
			{2,8,12}, {1,3,15}, {2,4,10}, {3,5,13}, {4,6,16}, {5,7,11}, {6,8,14}, {1,7,9}, {8,10,16}, {3,9,11}, {6,10,12}, {1,11,13}, {4,12,14}, {7,13,15}, {2,14,16}, {5,9,15}}),
		Plaquette("16 dots: (8,8) complete bipartite graph (BCC with PBC)", true, 1.0, vector<vector<int>> {
			{9,10,11,12,13,14,15,16}, {9,10,11,12,13,14,15,16}, {9,10,11,12,13,14,15,16}, {9,10,11,12,13,14,15,16}, {9,10,11,12,13,14,15,16}, {9,10,11,12,13,14,15,16}, {9,10,11,12,13,14,15,16}, {9,10,11,12,13,14,15,16},
			{1,2,3,4,5,6,7,8}, {1,2,3,4,5,6,7,8}, {1,2,3,4,5,6,7,8}, {1,2,3,4,5,6,7,8}, {1,2,3,4,5,6,7,8}, {1,2,3,4,5,6,7,8}, {1,2,3,4,5,6,7,8}, {1,2,3,4,5,6,7,8}}),
		Plaquette("16 dots: 3 adjacent septagons", false, 1.0, vector<vector<int>> { // index = 25
			{2,7,12}, {1,3,16}, {2,4}, {3,5}, {4,6}, {5,7}, {1,6,8}, {7,9}, {8,10}, {9,11}, {10,12}, {1,11,13}, {12,14}, {13,15}, {14,16}, {2,15}}),
		Plaquette("14 dots: septegon with 5-cycle cross-connections", false, 1.0, vector<vector<int>> {
			{2,5,11,14},{1,3},{2,4,7,13},{3,5},{1,4,6,9},{5,7},{3,6,8,11},{7,9},{5,8,10,13},{9,11},{1,7,10,12},{11,13},{3,9,12,14},{1,13}}),
		Plaquette("15 dots: 6 pentagons with cross-connections", false, 1.0, vector<vector<int>> {
			{2,5,9,14}, {1,3,7,12}, {2,4,10,15}, {3,5,8,13}, {1,4,6,11}, {5,7,15}, {2,6,8}, {4,7,9}, {1,8,10}, {3,9,11}, {5,10,12}, {2,11,13}, {4,12,14}, {1,13,15}, {3,6,14}}),
		Plaquette("15 dots: Extended Petersen-like graph", false, 1.0, vector<vector<int>> {
			{2,5,9}, {1,3,7}, {2,4,10}, {3,5,8}, {1,4,6}, {5,7,10,13}, {2,6,8,15}, {4,7,9,12}, {1,8,10,14}, {3,6,9,11}, {10,12,15}, {8,11,13}, {6,12,14}, {9,13,15}, {7,11,14}}),
		Plaquette("16 dots: octagon plaquette with 5-cycle connections", false, 1.0, vector<vector<int>> {
			{2,8,15},{1,3,11},{2,4,9},{3,5,13},{4,6,11},{5,7,15},{6,8,13},{1,7,9},{3,8,10,16},{9,11,14},{2,5,10,12},{11,13,16},{4,7,12,14},{10,13,15},{1,6,14,16},{9,12,15}})
	};

public:

	const int numvecs = 3;                            // Number of random vectors to sample
	const int decimalplaces = 4;                      // Number of decimal places for formatting output
	const string filename = "magnetoutput.txt";       // Text file name to write output to

	/*
	Randomly generated vectors are stored so that the same initial vectors can be used for multiple Plaquette calculations.
	randvecs[nd][ne] is the collection of all randomly-generated vectors for a number of quantum dots nd
	and number of electrons ne. Here ne runs from 2 to nd - 1 inclusive. For small values of nd or ne, randvecs is simply left empty.
	For a number of electrons greater than nd, instead use ne = 2 * nd - ne, and use a particle-hole transformation.
	Note, the particle-hole transformation requires that the Hamiltonian by multiplied by -1.
	randvecs[nd][ne][vi] is a vector<double> which represents the initial randomly-generated vector used for the calculation.
	vi runs from 0 to numvecs - 1 inclusive.
	randvecs[nd][ne][vi] has size binom(nd, ne)*binom(ne, ne / 2). Note: ne / 2 rounds down for odd ne.
	*/
	vector<vector<vector<vector<double>>>> randvecs;
	
	/*
	Calculating the SWAP matrices is a potentially expensive operation, so the SWAP matrices are stored after calculation
	so that other calculations which use the same number of dots can reuse these matrices.
	swapmats should be cleared after the matrices are no longer needed to free memory.
	*/
	vector<vector<SparseMatrix>> swapmats; 

	/*
	Calculates and caches binomial coefficients up to a max value maxn
	*/
	void initbinomial(int maxn) {
		binomialcache.reserve(maxn + 1);
		for (int n = static_cast<int>(binomialcache.size()); n <= maxn; n++) {
			binomialcache.emplace_back();
			binomialcache[n].reserve(n);
			binomialcache[n].push_back(1);
			for (int k = 1; k < n; k++) {
				binomialcache[n].push_back(binomialcache[n - 1][k - 1] + binomialcache[n - 1][k]);
			}
			if (n != 0) { binomialcache[n].push_back(1); }
		}
	}

	/*
	Returns the binomial coefficient n Choose k. Binomial coeffients are cached for fast access.
	Returns 0 if k < 0 or k > n
	*/
	int binom(int n, int k) {
		if (n >= static_cast<int>(binomialcache.size())) { initbinomial(n); }
		if (k < 0 || k > n) { return 0; }
		else { return binomialcache[n][k]; }
	}

	/*
	Normalizes the vector v (so that v dot v == 1), overwriting v with the output.
	*/
	void normalize(vector<double>& v) {
		double norm = 0;
		for (double d : v) {
			norm += d * d;
		}
		norm = sqrt(norm);
		for (int i = 0; i < static_cast<int>(v.size()); i++) {
			v[i] /= norm;
		}
	}

	/*
	Calculates the inner product: v1 dot v2.
	*/
	double dot(const vector<double>& v1, const vector<double>& v2) {
		double total = 0;
		for (int i = 0; i < static_cast<int>(v1.size()); i++) {
			total += v1[i] * v2[i];
		}
		return total;
	}

	/*
	dotarray(int, int, int, vector<DotState>) and dotarrayindex(vector<DotState>) define a basis for performing calculations.
	dotarray and dotarrayindex are inverse functions, that is:
	dotarrayindex(dotarray(index, nd, ne, nsd)); // returns index

	index is the index of the desired basis vector
	nd and ne are the total number of dots and electrons
	nsd is the total number of electrons with a down spin
	The basis vector is size nd, and is a list of the state of each dot for the desired basis vector.
	Note, the maximum index (size of the basis) is binom(nd, ne)*binom(ne, nsd)
	*/
	vector<DotState> dotarray(int index, int nd, int ne, int nsd) {
		vector<DotState> out(nd);
		int i = index;
		int sd_left = nsd; // number of down spins remaining
		int e_left = ne;   // number of electrons remaining
		for (int d = 0; d < nd; d++) {
			if (i < binom(nd - d - 1, e_left) * binom(e_left, sd_left)) {
				out[d] = DotState::EMPTY;
			}
			else {
				i -= binom(nd - d - 1, e_left) * binom(e_left, sd_left);
				if (i < binom(nd - d - 1, e_left - sd_left - 1) * binom(nd - d - e_left + sd_left, sd_left)) {
					out[d] = DotState::UP;
					e_left--;
				}
				else {
					out[d] = DotState::DOWN;
					i -= binom(nd - d - 1, e_left - sd_left - 1) * binom(nd - d - e_left + sd_left, sd_left);
					e_left--;
					sd_left--;
				}
			}
		}
		return move(out);
	}

	/*
	dotarray(int, int, int, vector<DotState>) and dotarrayindex(vector<DotState>) define a basis for performing calculations.
	dotarray and dotarrayindex are inverse functions, that is:
	dotarrayindex(dotarray(index, nd, ne, nsd)); // returns index

	dotarr is a vector<DotState> specifying the state of each quantum dot
	dotarrayindex(dotarr) returns the corresponding index for that basis vector
	Note, the maximum index (size of the basis) is binom(nd, ne)*binom(ne, nsd)
	*/
	int dotarrayindex(const vector<DotState>& dotarr) {
		int out = 0;
		int nd = dotarr.size();
		int e_left = 0;  // number of electrons remaining
		int sd_left = 0; // number of down spins remaining
		for (DotState ds : dotarr) {
			if (ds == DotState::UP) { e_left++; }
			else if (ds == DotState::DOWN) { e_left++; sd_left++; }
		}
		for (int d = 0; d < nd; d++) {
			if (dotarr[d] == DotState::DOWN) {
				out += binom(nd - d - 1, e_left) * binom(e_left, sd_left) +
					binom(nd - d - 1, e_left - sd_left - 1) * binom(nd - d - e_left + sd_left, sd_left);
				e_left--;
				sd_left--;
			}
			else if (dotarr[d] == DotState::UP) {
				out += binom(nd - d - 1, e_left) * binom(e_left, sd_left);
				e_left--;
			}
		}
		return out;
	}

	/*
	Calculates eigenvallues of a symmetric matrix using Jacobi's method.
	matrix.size() must be equal to matrix[0].size, the dimension of the matrix.
	WARNING: matrix must be symmetric matrix.
	WARNING: Do not use this method for vary large matrices. Instead use the power-iteration method.
	*/
	vector<double> eigvalsjacobi(vector<vector<double>> matrix, double error) { // matrix is passed by value since it will be edited
		int n = matrix.size();
		int state = n;
		int m, l, j;
		double p, y, d, r, c, s, t, atemp;
		double errn = error / n;
		vector<double> out;
		vector<int> ind;
		vector<bool> ischanged;
		out.reserve(n);
		ind.reserve(n);
		ischanged.reserve(n);
		for (int k = 0; k < n; k++) {
			out.push_back(matrix[k][k]);
			ischanged.push_back(true);
			j = k + 1;
			for (int i = k + 2; i < n; i++) { if (abs(matrix[k][i]) > abs(matrix[k][j])) { j = i; } }
			ind.push_back(j);
		}
		while (state != 0) {
			m = 0;
			for (int k = 1; k < n - 1; k++) {
				if (abs(matrix[k][ind[k]]) > abs(matrix[m][ind[m]])) { m = k; }
			}
			l = ind[m];
			p = matrix[m][l];
			if (abs(p) <= errn) {
				state = 0;
			}
			else {
				y = (out[l] - out[m]) / 2;
				d = abs(y) + sqrt(p * p + y * y);
				r = sqrt(p * p + d * d);
				c = d / r;
				s = p / r;
				t = p * p / d;
				if (y < 0.0) { s = -s; t = -t; }
				matrix[m][l] = 0.0;
				out[m] -= t;
				if (abs(t) < error && ischanged[m]) {
					ischanged[m] = false;
					state--;
				}
				else if (abs(t) >= error && !ischanged[m]) {
					ischanged[m] = true;
					state++;
				}
				out[l] += t;
				if (abs(t) < error && ischanged[l]) {
					ischanged[l] = false;
					state--;
				}
				else if (abs(t) >= error && !ischanged[l]) {
					ischanged[l] = true;
					state++;
				}
				for (int i = 0; i < m; i++) {
					atemp = matrix[i][m];
					matrix[i][m] = c * matrix[i][m] - s * matrix[i][l];
					matrix[i][l] = s * atemp + c * matrix[i][l];
				}
				for (int i = m + 1; i < l; i++) {
					atemp = matrix[m][i];
					matrix[m][i] = c * matrix[m][i] - s * matrix[i][l];
					matrix[i][l] = s * atemp + c * matrix[i][l];
				}
				for (int i = l + 1; i < n; i++) {
					atemp = matrix[m][i];
					matrix[m][i] = c * matrix[m][i] - s * matrix[l][i];
					matrix[l][i] = s * atemp + c * matrix[l][i];
				}
				j = m + 1;
				for (int i = m + 2; i < n; i++) { if (abs(matrix[m][i]) > abs(matrix[m][j])) { j = i; } }
				ind[m] = j;
				for (int i = 0; i < m; i++) { if (abs(matrix[i][m]) > abs(matrix[i][ind[i]])) { ind[i] = m; } }
				j = l + 1;
				for (int i = l + 2; i < n; i++) { if (abs(matrix[l][i]) > abs(matrix[l][j])) { j = i; } }
				ind[l] = j;
				for (int i = 0; i < l; i++) { if (abs(matrix[i][l]) > abs(matrix[i][ind[i]])) { ind[i] = l; } }
			}
		}
		for (int k = 0; k < n - 1; k++) {
			m = k;
			for (int l = k + 1; l < n; l++) {
				if (out[l] > out[m]) { m = l; }
			}
			if (k != m) {
				atemp = out[m];
				out[m] = out[k];
				out[k] = atemp;
			}
		}
		return move(out);
	}

	/*
	Calculates the SWAP matrices for a specified number of dots nd,
	and initializes swapmats[nd][ne] for ne from 2 to nd - 1.
	*/
	void initswapmat(int nd) {
		if (static_cast<int>(swapmats.size()) < nd + 1) { swapmats.resize(nd + 1); }
		swapmats[nd].resize(nd);
		for (int ne = 2; ne < nd; ne++) {
			int n = binom(nd, ne) * binom(ne, ne / 2);
			SparseMatrix& out = swapmats[nd][ne];
			out.clear(n);
			vector<DotState> da;
			int kswap;
			for (int k = 0; k < n; k++) {
				da = dotarray(k, nd, ne, ne / 2);
				for (int i = 0; i < nd - 1; i++) {
					if (da[i] == DotState::UP) {
						da[i] = DotState::DOWN;
						for (int j = i + 1; j < nd; j++) {
							if (da[j] == DotState::DOWN) {
								da[j] = DotState::UP;
								kswap = dotarrayindex(da);
								out.add(k, kswap, 1.0);
								out.add(kswap, k, 1.0);
								da[j] = DotState::DOWN;
							}
						}
						da[i] = DotState::UP;
					}
				}
			}
		}
	}

	/*
	Clears swapmats[nd] to free up memory when it is no longer needed.
	*/
	void clearswapmat(int nd) {
		if (nd < static_cast<int>(swapmats.size())) { swapmats[nd].clear(); }
	}

	/*
	Generates random vectors for a specified number of dots nd,
	and initializes randvecs[nd][ne][vi] for ne from 2 to nd - 1, and vi from 0 to numvecs - 1.
	*/
	void initvecs(int nd) {
		if (static_cast<int>(randvecs.size()) < nd + 1) { randvecs.resize(nd + 1); }
		randvecs[nd].resize(nd);
		for (int ne = 2; ne < nd; ne++) {
			randvecs[nd][ne].resize(numvecs);
			for (int vi = 0; vi < numvecs; vi++) {
				int n = binom(nd, ne) * binom(ne, ne / 2);
				randvecs[nd][ne][vi].clear();
				randvecs[nd][ne][vi].reserve(n);
				for (int i = 0; i < n; i++) {
					randvecs[nd][ne][vi].push_back(normdist(rgen));
				}
				normalize(randvecs[nd][ne][vi]);
			}
		}
	}

	/*
	Clears randvecs[nd] to free up memory when they are no longer needed.
	*/
	void clearvecs(int nd) {
		if (nd < static_cast<int>(randvecs.size())) { randvecs[nd].clear(); }
	}

	/*
	Projects a vector v into the total-spin-J subspace and normalizes the result, overwriting v with the output.
	v is written in the basis defined by dotarray(index, nd, ne, ne / 2, dotarr).
	v has size binom(nd, ne) * binom(ne, ne / 2).
	nd and ne are the number of dots and electrons
	j2 is 2*J, where J is the desired spin subspace to project v into
	WARNING: initswapmat(nd) must be called before calling this method
	*/
	void projectvec(vector<double>& v, int nd, int ne, int j2) {
		vector<double> w;
		SparseMatrix& sm = swapmats[nd][ne];
		int n = binom(nd, ne) * binom(ne, ne / 2);
		for (int jj = ne % 2; jj <= ne; jj += 2) {
			if (jj != j2) {
				w = sm.times(v);
				double swapeig = static_cast<double> ((jj * (jj + 2) - (ne % 2) - 2 * ne) / 4);
				for (int i = 0; i < n; i++) {
					w[i] -= swapeig * v[i];
				}
				normalize(w);
				v = move(w);
			}
		}
	}

	/*
	Builds the Hamiltonian in the basis defined by dotarray(index, h.size(), ne, ne / 2).
	h is the single-particle Hamiltonian, and the number of dots is defined by h.size()
	ne is the number of electrons
	*/
	SparseMatrix buildh0(const vector<vector<double>>& h, int ne) {
		int nd = h.size();
		int n = binom(nd, ne) * binom(ne, ne / 2);
		SparseMatrix out(n);
		vector<DotState> da;
		DotState ds;
		int kswap;
		for (int k = 0; k < n; k++) {
			double total = 0.0;
			da = dotarray(k, nd, ne, ne / 2);
			for (int i1 = 0; i1 < nd; i1++) {
				if (da[i1] != DotState::EMPTY) {
					total += h[i1][i1];
					ds = da[i1];
					da[i1] = DotState::EMPTY;
					bool iseven = true;
					for (int i0 = i1 + 1; i0 < nd; i0++) {
						if (da[i0] == DotState::EMPTY) {
							da[i0] = ds;
							kswap = dotarrayindex(da);
							if (h[i1][i0] != 0.0) {
								if (iseven) {
									out.add(k, kswap, h[i1][i0]);
									out.add(kswap, k, h[i0][i1]);
								}
								else {
									out.add(k, kswap, -h[i1][i0]);
									out.add(kswap, k, -h[i0][i1]);
								}
							}
							da[i0] = DotState::EMPTY;
						}
						else {
							iseven = !iseven;
						}
					}
					da[i1] = ds;
				}
			}
			out.add(k, k, total);
		}
		return move(out);
	}

	/*
	Uses the power iteration method to find the lowest-energy eigenvalue of the Hamiltonian for a given spin J.

	The power iteration method naturally converges to the eigenstate with the greatest absolute value.
	However, it is possible that this eigenvalue is positive.
	In order to ensure that a negative energy is found by the method,
	offset * 1 is subtracted from h0 before finding eigenvalues (where 1 is the identity matrix).
	offset is readded to the energy at the end of the method to ensure the proper energy is given.
	It is the user's responsibility to specify an offset such that |E_min - offset| > |E_max - offset|.

	This method uses vectors which are written in a basis which includes states outside the desired subspace (spin = J).
	If all mathematical operations are done exactly, all that is required is to project the vectors once into the desired subspace.
	However, due to machine-level arithmetic imprecisions, small components of the vector which lie outside of the subspace can remain.
	Once intoduced, these errors will grow if their corresponding eigenvalues have larger magnitude than the target eigenvalue.
	Thus, every so often, the vector should be reprojected into the spin-J subspace to eliminate these errors.
	projfreq is the number of iterations performed before projecting back into the spin-J subspace.

	h0 is the Hamiltonian as output by buildh0(h, ne)
	v0s is a list of random normalized initial vectors
	err is the error cutoff, which determines the precision of the result
	nd and ne are the number of dots and electrons
	j2 is 2*J, where J is the desired spin
	WARNING: v0s[i] must be normalized
	*/
	double poweriter(const SparseMatrix& h0, const vector<vector<double>>& v0s, double offset, double err, int nd, int ne, int j2, int projfreq) {
		bool isdone = false;
		int itr = 0;
		double err2 = err * err;
		vector<vector<double>> vs = v0s;
		int nv = vs.size();
		int n = h0.size();
		vector<double> w;
		vector<double> lambda(nv, 0.0);
		double lmax;
		while (!isdone) {
			if (itr == 0) {
				itr = projfreq;
				for (vector<double>& v : vs) {
					projectvec(v, nd, ne, j2);
				}
			}
			itr--;
			double delta = 0.0;
			lmax = 0.0;
			for (int vi = 0; vi < nv; vi++) {
				w = h0.times(vs[vi]);
				for (int i = 0; i < n; i++) {
					w[i] -= offset * vs[vi][i];
				}
				double l = dot(w, vs[vi]);
				if (abs(l) > abs(lmax)) { lmax = l; delta = abs(l - lambda[vi]); }
				lambda[vi] = l;
				normalize(w);
				vs[vi] = move(w);
			}
			if (delta <= err2) { isdone = true; }
		}
		return lmax + offset;
	}

	/*
	Returns the single-particle Hamiltonian for plaquettes[index].
	if isHole is true, the Hamiltonian is multiplied by -1 to account for particle-hole transformation
	*/
	SparseMatrix geth(int index, bool isHole) {
		SparseMatrix out;
		if (index >= 0 && index < static_cast<int>(plaquettes.size())) {
			out = plaquettes[index].h;
			if (isHole) { out.multiplyby(-1.0); }
		}
		return move(out);
	}

	/*
	Returns the name of plaquettes[index].
	*/
	string getlabel(int index) {
		if (index >= 0 && index < static_cast<int>(plaquettes.size())) { return plaquettes[index].name; }
		return "invalid index";
	}

	/*
	Returns the number of quantum dots in plaquettes[index].
	*/
	int getsize(int index) {
		if (index >= 0 && index < static_cast<int>(plaquettes.size())) { return plaquettes[index].size(); }
		return 0;
	}

	/*
	Returns plaquettes[index].isBipartite
	*/
	bool getBipartite(int index) {
		if (index >= 0 && index < static_cast<int>(plaquettes.size())) { return plaquettes[index].isBipartite; }
		return false;
	}

	/*
	Formats a double to string
	*/
	string formatdouble(double d) {
		stringstream ss;
		ss.setf(ios::fixed);
		ss.precision(decimalplaces);
		ss << d;
		string s = ss.str();
		int i = s.size() - 1;
		while (s[i] == '0') { i--; }
		if (s[i] == '.') { i--; }
		s.erase(i + 1);
		return move(s);
	}

	/*
	Formats an integer number of seconds as a string with hours, mins, secs.
	Ex: formattime(12345); // returns "3h 25m 45s"
	*/
	string formattime(int numsecs) {
		return to_string(numsecs / 3600) + "h " + to_string((numsecs / 60) % 60) + "m " + to_string(numsecs % 60) + "s";
	}

	/*
	Formats results as a string of LaTeX code which displays the results nicely in a table.
	The ground state in each row is marked in bold.
	
	results is a 2d vector which contains the ground state-energies output by poweriter(...)
	results[ne][j] is the ground-state energy for a number of electrons ne, where ne runs from 2 to nd-1,
	and spin j = J (if J is an integer), or j = J - 1/2 (if J is a half-integer), here j runs from 0 to ne/2

	nd is the number of dots
	label is the name of the plaquette
	time_h is the number of seconds taken to run the hole part of the calculation (if applicable)
	time_e is the number of seconds taken to run the electron part of the calculation
	isBipartite is true if the plaquette is bipartite, or false otherwise
	isHole is true if results corresponds to the holes, or false if results corresponds to electrons
	*/
	string formatresults(const vector<vector<double>>& results, int nd, const string& label, int time_h, int time_e, bool isBipartite, bool isHole) {
		stringstream ss;
		double err = pow(10.0, -decimalplaces);
		int ncol = (nd + 1) / 2; // not including header
		if (isBipartite || !isHole) {
			ss << "hole doped time: " << formattime(time_h) << "\n";
			if (!isBipartite) {
				ss << "electron doped time: " << formattime(time_e) << "\n";
			}
			ss << "\\begin{tabular}{|";
			for (int i = 0; i < ncol + 1; i++) { ss << "c|"; }
			ss << "}\n\\hline\n"
				<< "\\multicolumn{" << ncol + 1 << "}{|c|}{" << label << "}\\\\\\hline\n"
				<< "& \\multicolumn{" << ncol << "}{|c|}{Spin}\\\\\\hline\n"
				<< "\\# of el.";
			for (int i = 0; i < ncol; i++) {
				ss << " & " << i;
				if (i < ncol - 1 || nd % 2 == 0) { ss << ", " << (2 * i + 1) << "/2"; }
			}
			ss << "\\\\\\hline\n";
		}
		int nes, nec, nei;
		if (isBipartite || isHole) {
			nes = nd - 1;
			nec = 1;
			nei = -1;
		}
		else {
			nes = 2;
			nec = nd;
			nei = 1;
		}
		for (int ne = nes; ne != nec; ne += nei) {
			double smallest = 0.0;
			if (isBipartite) {
				ss << ne << ", " << (2 * nd - ne) << " ";
			}
			else if (isHole) {
				ss << ne << " ";
			}
			else {
				ss << (2 * nd - ne) << " ";
			}
			int ndata = ne / 2 + 1;
			for (int i = 0; i < ndata; i++) {
				if (results[ne][i] < smallest) { smallest = results[ne][i]; }
			}
			for (int i = 0; i < ndata; i++) {
				if (results[ne][i] - smallest <= err) {
					ss << "& {\\bf " << formatdouble(results[ne][i]) << "t} ";
				}
				else {
					ss << "& " << formatdouble(results[ne][i]) << "t ";
				}
			}
			for (int i = ndata; i < ncol; i++) { ss << "&"; }
			ss << "\\\\\\hline\n";
		}
		if (isBipartite || isHole) {
			ss << "\\end{tabular}\n\n";
		}
		else {
			ss << nd << " ";
			for (int i = 0; i < ncol; i++) { ss << "& 0 "; }
			ss << "\\\\\\hline\n";
		}
		return ss.str();
	}

	/*
	Runs the program for the plaquettes indicated by indeces.
	*/
	void run(const vector<int>& indecies) {
		vector<vector<double>> resultsh, resultse, hnorm;
		int time_h, time_e;
		ofstream file;
		for (int ii = 0; ii < static_cast<int>(indecies.size()); ii++) {
			int index = indecies[ii];
			cout << "index: " << index;
			auto starttime = chrono::high_resolution_clock::now();
			SparseMatrix h = geth(index, true);
			hnorm = h.normalform();
			int nd = getsize(index);
			bool isBipartite = getBipartite(index);
			int ndprev = 0;
			if (ii != 0) { ndprev = getsize(indecies[ii - 1]); };
			if (ndprev != nd || ii == 0) {
				clearswapmat(ndprev);
				initswapmat(nd);
				clearvecs(ndprev);
				initvecs(nd);
			}
			resultsh.resize(nd);
			resultse.resize(nd);
			vector<double> eigs = eigvalsjacobi(hnorm, 0.01);
			for (int ne = nd - 1; ne >= 2; ne--) {
				SparseMatrix h0;
				double offset = 0.0;
				resultsh[ne].clear();
				for (int i = 0; i < ne; i++) { offset += eigs[i]; }
				h0 = buildh0(hnorm, ne);
				for (int j2 = ne % 2; j2 <= ne; j2 += 2) {
					resultsh[ne].push_back(poweriter(h0, randvecs[nd][ne], offset / 2.0, pow(10.0, -decimalplaces), nd, ne, j2, 100));
				}
			}
			auto endtime = chrono::high_resolution_clock::now();
			time_h = chrono::duration_cast<chrono::seconds>(endtime - starttime).count();
			cout << "; hole time: " << formattime(time_h);
			if (!isBipartite) {
				starttime = chrono::high_resolution_clock::now();
				h.multiplyby(-1.0);
				hnorm = h.normalform();
				for (int ne = nd - 1; ne >= 2; ne--) {
					SparseMatrix h0;
					double offset = 0.0;
					resultse[ne].clear();
					for (int i = 0; i < ne; i++) { offset -= eigs[nd - 1 - i]; }
					h0 = buildh0(hnorm, ne);
					for (int j2 = ne % 2; j2 <= ne; j2 += 2) {
						resultse[ne].push_back(poweriter(h0, randvecs[nd][ne], offset / 2.0, pow(10.0, -decimalplaces), nd, ne, j2, 100));
					}
					h0.clear(0);
				}
				endtime = chrono::high_resolution_clock::now();
				time_e = chrono::duration_cast<chrono::seconds>(endtime - starttime).count();
				cout << "; el time: " << formattime(time_h);
			}
			string stre = "";
			string strh = formatresults(resultsh, nd, getlabel(index), time_h, time_e, isBipartite, true);
			if (!isBipartite) { stre = formatresults(resultse, nd, getlabel(index), time_h, time_e, false, false); }
			file.open(filename, ios_base::app);
			file << stre << strh;
			file.close();
			cout << "\n";
		}
	}


};

/*
Runs the program. Command line parameters are ignored.
To indicate which Plaquettes to run, change the vector<int> passed to m.run(indeces)
*/
int main(int argc, char** argv) {
	Main m;
	m.run(vector<int> {0, 1, 2, 3, 4, 5}); // these plaquettes are run
	return 0;
}
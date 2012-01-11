/** @file
 * This are various variations of a dslash kernel based on SOA storage.
 * The main kernel of which the other are variants is dslash_eoprec.
 *
 * The file is rather lenghty because it contains multiple kernels and
 * all things used by the kernels. The code of a single kernel is still
 * long, but keeping the structure of the file in mind can "easily" be read.
 *
 * The structure of the file is as follows.
 *  * General Definitions, e.g. defining problem dimenstions and array strides
 *  * Data types representing physical values
 *  * Data types used to address values (index types)
 *  * Functions used for working with index types
 *  * Functions implementing operations for the physical values
 *  * Variations of the part of the kernel working in one direction
 *  * Kernels
 *
 * Please not that the APP Kernel Analyzer (Dec 2011) gives strange readings
 * for the resource usage of theses kernels. You can use
 * https://github.com/theMarix/pyclKernelAnalyzer to get values corresponding
 * to actual values observed in the application using theses kernels.
 *
 * Snapshot values for Catalyst 11.11 (APP 2.5) measures using
 * pyclKernelAnalyzer/analyze.py -d 0 SoA.cl
 *
 * Kernel Name, GPRs Scratch Registers, Local Memory (Bytes), Device Version, Driver Version
 * slash_eoprec_1dir                       49       0       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec_2dirs                     62       0       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec_3dirs                     62      10       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec                           62      12       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec_lim_group_size            71       0       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec_simplified                54       0       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec_simplified_loop           62       7       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec_simplified_loop_unrolled  54       0       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec_simplified_loop_nounroll  62       7       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 * dslash_eoprec_simplified_loop_noret     52       0       0       OpenCL 1.1 AMD-APP-SDK-v2.5 (793.1)     CAL 1.4.1607
 *
 * To give a briew overview what happening here. The kernel operations on a four-dimensional lattice. Each site
 * of the lattice is an element of type spinor. These sites are linked with their nearest neighbours via elements
 * of type Matrixsu3. The whole problem is red-black conditioned. So the input contains only black, the output only
 * red elements or vice versa. In this code we call this even-odd.
 *
 * On run of the kernel calculates a new set of spinors. Each site is basically calculated by summing up the
 * nearest neighbours in all four directions (time, x, y, z) - which is 8 values (forward and backward) weighted by
 * the linking Matrixsu3 element. There is some additional foo based on the direction, which is why we actually
 * need an own function for each direction: dslash_eoprec_local0/1/2/3. For comparison there is also are also kernels
 * which, physically incorrectly, perform the same calculation regardless of direction (it's basically the x-direction function):
 * dslash_eoprec_local. This is used by the simplified kernels.
 *
 * Looking at the resource utilization above two points stick out:
 *  1. The dslash_eoprec_3dirs kernel uses 10 scratch registers where the dslash_eoprec_2dirs kernel uses none.
 *  2. In the simplified case unrolled loops use less registers than non-unrolled. With the notable exception of the non-unrolled
 *     version where the return value was optimized out by using a pointer. This is especially strange, as pointers to registers
 *     often make the register count explode.
 *     The reason it wonders me the the register optimization is better in the unrolled case is, that in that case the simplification
 *     should not produce any other code than the original version. With the exception of some additions being subtractions or vice versa.
 */

//
// General Definitions, e.g. defining problem dimenstions and array strides
//

#ifdef cl_khr_fp
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

//-DNSPACE=16 -DNTIME=12 -DVOLSPACE=4096 -DVOL4D=49152 -DSU3SIZE=9 -DSTAPLEMATRIXSIZE=9 -D_USEDOUBLEPREC_ -D_DEVICE_DOUBLE_EXTENSION_KHR_ -D_USEGPU_ -DGAUGEFIELD_STRIDE=196672 -I/home/bach/QCD/clhmc/prog -D_FERMIONS_ -DSPINORSIZE=12 -DHALFSPINORSIZE=6 -DSPINORFIELDSIZE=49152 -DEOPREC_SPINORFIELDSIZE=24576
//-D_TWISTEDMASS_ -DKAPPA=0.1 -DMKAPPA=-0.1 -DKAPPA_SPATIAL_RE=0.1 -DMKAPPA_SPATIAL_RE=-0.1 -DKAPPA_SPATIAL_IM=0 -DMKAPPA_SPATIAL_IM=-0 -DKAPPA_TEMPORAL_RE=0.0965926 -DMKAPPA_TEMPORAL_RE=-0.0965926 -DKAPPA_TEMPORAL_IM=0.0258819 -DMKAPPA_TEMPORAL_IM=-0.0258819 -DEOPREC_SPINORFIELD_STRIDE=24640
//-DMU=0.005 -DMUBAR=0.001 -DMMUBAR=-0.001
#define NSPACE 16
#define NTIME 12
#define VOLSPACE 4096
#define VOL4D 49152
#define _FERMIONS_
#define _TWISTEDMASS_
#define KAPPA_TEMPORAL_RE 0.0965926
#define MKAPPA_TEMPORAL_RE -0.0965926
#define KAPPA_TEMPORAL_IM 0.0258819
#define MKAPPA_TEMPORAL_IM -0.0258819
#define KAPPA_SPATIAL_RE 0.1
#define MKAPPA_SPATIAL_RE -0.1
#define KAPPA_SPATIAL_IM 0
#define MKAPPA_SPATIAL_IM -0
#define EOPREC_SPINORFIELDSIZE 24576
#define GAUGEFIELD_STRIDE 196672
#define EOPREC_SPINORFIELD_STRIDE 24640

/** Number of dimensions of the lattice */
#define NDIM 4

#define EVEN 0
#define ODD 1

//
// Data types representing physical values
//

typedef double hmc_float __attribute__((aligned(8)));

typedef struct {
	hmc_float re;
	hmc_float im;
} hmc_complex
__attribute__((aligned(16)));

//CP: new defs for spinors
typedef struct {
	hmc_complex e0;
	hmc_complex e1;
	hmc_complex e2;
} su3vec
__attribute__((aligned(16)));

typedef struct {
	su3vec e0;
	su3vec e1;
	su3vec e2;
	su3vec e3;
} spinor
__attribute__((aligned(32)));

typedef struct {
	hmc_complex e00;
	hmc_complex e01;
	hmc_complex e02;
	hmc_complex e10;
	hmc_complex e11;
	hmc_complex e12;
	hmc_complex e20;
	hmc_complex e21;
	hmc_complex e22;
} Matrixsu3;

__constant hmc_complex hmc_complex_zero = {0., 0.};

//
// Data types used to address values (index types)
//

/** @file
 * Device code for lattice geometry handling
 * Geometric conventions used should only be applied in this file
 * All external functions should only use functions herein!!!
 */

////////////////////////////////////////////////////////////////
// General Definitions
////////////////////////////////////////////////////////////////

/** In the following, identify (coord being coord_full or coord_spatial):
 *coord.x == x
 *coord.y == y
 *coord.z == z
 *(coord.w == t)
 * NOTE: This does not necessarily reflect the geometric conventions used!
 */

/** index type to store all spacetime coordinates  */
typedef uint4 coord_full;

/** index type to store all spatial coordinates */
typedef uint3 coord_spatial;

/** index type to store a temporal coordinate */
typedef uint coord_temporal;

/** index type to store a site/link position */
typedef uint site_idx;
typedef uint link_idx;

/** index type to store a spatial site position */
typedef uint spatial_idx;

/** index type to store a direction (0..3) */
typedef uint dir_idx;

/** An index type storing space (index) and time (coordinate) seperate. */
typedef struct {
	spatial_idx space;
	coord_temporal time;
} st_idx;

////////////////////////////////////////////////////////////////
// Geometric Conventions
////////////////////////////////////////////////////////////////

/** Identify each spacetime direction */
#define TDIR 0
#define XDIR 1
#define YDIR 2
#define ZDIR 3

//
// Functions used for working with index types
//

/**
 * The following conventions are used:
 * (NS: spatial extent, NT: temporal extent, NDIM: # directions, VOL4D: lattice volume, VOLSPACE: spatial volume)
 * A spatial idx is adressed as spatial_idx(x,y,z) = x * NS^(XDIR-1) + y * NS^(YDIR-1) + z * NS^(ZDIR-1)
 * A site idx is addressed as site_idx(x,y,z,t) = spatial_idx(x,y,z) + t*NS*NS*NS
 * A link idx is addressed as link_idx(x,y,z,t,mu) = mu + NDIM*site_idx(x,y,z,t)
 */

/**
 * The following conventions are used in tmlqcd:
 * #define TDIR 0
 * #define XDIR 1
 * #define YDIR 2
 * #define ZDIR 3    (same)
 * A spatial idx is adressed as spatial_idx(x,y,z) = x * NS^(3-XDIR) + y * NS^(3-YDIR) + z * NS^(3-ZDIR) (different)
 * A site idx is addressed as site_idx(x,y,z,t) = spatial_idx(x,y,z) + t * VOLSPACE (same)
 * A link idx is addressed as an array with 2 indices.
 * @NOTE: This is more consistent then our convention because it yields
 *    site_idx = x * NS^(3-XDIR) + y * NS^(3-YDIR) + z * NS^(3-ZDIR) + t * NS^(3-TDIR)
 * whereas ours gives
 *    site_idx = x * NS^(XDIR-1) + y * NS^(YDIR-1) + z * NS^(ZDIR-1) + t * NS^(3-TDIR)
 */

////////////////////////////////////////////////////////////////
// Functions relying explicitely on the geometric conventions
// defined above (w/o even-odd functions!!)
////////////////////////////////////////////////////////////////

/**
 * with this set to false or true, one can switch between our original convention and
 * the one from tmlqcd.
 * our original:
 * spatial_idx = x + NS * y + NS*NS * z
 * tmlqcd:
 * spatial_idx = z + NS * y + NS*NS * x
 * NOTE: the ifs and elses used here should be removed by the compiler
 *       Nevertheless, one could also change to a permanent convention here
 */
#define TMLQCD_CONV false

/** spatial coordinates <-> spatial_idx
 *@todo this can be generalize using the definitions of the spatial directions
 *see  http://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
 */
spatial_idx get_spatial_idx(const coord_spatial coord)
{
	bool tmp = TMLQCD_CONV;
	if( tmp ) {
		return (coord.z +  NSPACE * coord.y + NSPACE * NSPACE * coord.x);
	} else {
		return (coord.x +  NSPACE * coord.y + NSPACE * NSPACE * coord.z);
	}
}
coord_spatial get_coord_spatial(const spatial_idx nspace)
{
	coord_spatial coord;
	bool tmp = TMLQCD_CONV;
	if( tmp ) {
		coord.x = nspace / NSPACE / NSPACE;
		uint acc = coord.x;
		coord.y = nspace / NSPACE - NSPACE * acc;
		acc = NSPACE * acc + coord.y;
		coord.z = nspace - NSPACE * acc;
	} else {
		coord.z = nspace / NSPACE / NSPACE;
		uint acc = coord.z;
		coord.y = nspace / NSPACE - NSPACE * acc;
		acc = NSPACE * acc + coord.y;
		coord.x = nspace - NSPACE * acc;
	}
	return coord;
}

/**
 * st_idx <-> site_idx using the convention:
 * site_idx = x + y*NS + z*NS*NS + t*NS*NS*NS
 * = site_idx_spatial + t*VOLSPACE
 * <=>t = site_idx / VOLSPACE
 * site_idx_spatial = site_idx%VOLSPACE
 */
site_idx get_site_idx(const st_idx in)
{
	return in.space + VOLSPACE * in.time;
}
st_idx get_st_idx_from_site_idx(const site_idx in)
{
	st_idx tmp;
	tmp.space = in % VOLSPACE;
	tmp.time = in / VOLSPACE;
	return tmp;
}

/** returns eo-vector component from st_idx */
site_idx get_eo_site_idx_from_st_idx(st_idx in);

site_idx get_link_idx_SOA(const dir_idx mu, const st_idx in)
{
	const uint3 space = get_coord_spatial(in.space);
	// check if the site is odd (either spacepos or t odd)
	// if yes offset everything by half the number of sites (number of sites is always even...)
	site_idx odd = (space.x ^ space.y ^ space.z ^ in.time) & 0x1 ? (VOL4D / 2) : 0;
	return mu * VOL4D + odd + get_eo_site_idx_from_st_idx(in);
}

/**
 * (st_idx, dir_idx) -> link_idx and
 * link_idx -> st_idx, respectively.
 *using the convention:
 *link_idx = mu + NDIM * site_idx
 */
link_idx get_link_idx(const dir_idx mu, const st_idx in)
{
	return mu + NDIM * get_site_idx(in);
}
st_idx get_st_idx_from_link_idx(const link_idx in)
{
	st_idx tmp;
	site_idx idx_tmp = in / NDIM;
	tmp = get_st_idx_from_site_idx(idx_tmp);
	return tmp;
}

////////////////////////////////////////////////////////////////
// Accessing the lattice points
////////////////////////////////////////////////////////////////

/** returns all coordinates from a given site_idx */
coord_full get_coord_full(const site_idx in)
{
	st_idx tmp;
	tmp = get_st_idx_from_site_idx(in);
	coord_spatial tmp2;
	tmp2 = get_coord_spatial(tmp.space);
	coord_full res;
	res.w = tmp.time;
	res.x = tmp2.x;
	res.y = tmp2.y;
	res.z = tmp2.z;
	return res;
}

////////////////////////////////////////////////////////////////
// Even/Odd functions
// NOTE: These explicitely rely on the geometric conventions
// and should be reconsidered if the former are changed!!
////////////////////////////////////////////////////////////////

/**
 * Even-Odd preconditioning means applying a chessboard to the lattice,
 * and distinguish between black and white (even/odd) sites.
 * this can be done in a 2D plane. Going to 4D then means to repeat this
 * 2D plane alternating even and odd sites.
 *
 * Schematically one has (assume 4x4 plane, x: even, o: odd):
 * oxox
 * xoxo
 * oxox
 * xoxo
 *
 * Take a look at the coordinates (call them w (ordinate) and t (abscise) ):
 * w/t | 0 | 1 | 2 | 3 |
 *   0 | x | o | x | o |
 *   1 | o | x | o | x |
 *   2 | x | o | x | o |
 *   3 | o | x | o | x |
 *
 * Assume that a coordinate in the plane is calculated as pos = t + w*4
 * Then, the coordinates of the even and odd sites are:
 * even|odd
 * 0|1
 * 2|3
 * 5|4
 * 7|6
 * 9|8
 * 11|10
 * 12|13
 * 14|15
 *
 * One can see a regular pattern, which can be described by the functions
 * f_even:2*w + int(2w/4)
 * f_odd: 1 + 2*w - int(2w/4)
 * Thus, given an index pair (w,t) one can map it to an even or odd site
 * by pos = f_even/odd(w) + 4*2*t, assuming that t runs only from 0 to 2
 * (half lattice size).
 * Extending this to 4D (call additional coordinates g1,g2) can be done
 * by alternating between f_even and f_odd, depending on g1 and g2.
 * E.g. one can do that via (g1 + g2)%2 and (g1 + g2 + 1)%2 prefactors.
 * Extending these consideratios to arbitrary lattice extend is straightforward.
 * NOTE: In general, a 4D site is even or odd if (x+y+z+t)%2 is 0 or 1.
 * However, in general one wants to have a loop over all even or odd sites.
 * Then this condition is not applicable, if not one loops over ALL sites and
 * just leaves out the ones where the condition is not fulfilled.
 * Since we do not want to create a possibly big table to map loop-variable
 * -> even/odd site we will use the pattern above for that!
 */

/** returns eo-vector component from st_idx */
site_idx get_eo_site_idx_from_st_idx(st_idx in)
{
	return get_site_idx(in) / 2;
}

/** the functions mentioned above
 * @todo this may be generalized because it relys on the current geometric conventions!!
 */
site_idx calc_even_spatial_idx(coord_full in)
{
	bool switcher = TMLQCD_CONV;
	if(switcher) {
		return
		  (uint)((in.x + in.w    ) % 2) * (1 + 2 * in.z - (uint) (2 * in.z / NSPACE)) +
		  (uint)((in.w + in.x + 1) % 2) * (    2 * in.z + (uint) (2 * in.z / NSPACE)) +
		  2 * NSPACE * in.y +
		  NSPACE * NSPACE * in.x;
	} else {
		return
		  (uint)((in.z + in.w    ) % 2) * (1 + 2 * in.x - (uint) (2 * in.x / NSPACE)) +
		  (uint)((in.w + in.z + 1) % 2) * (    2 * in.x + (uint) (2 * in.x / NSPACE)) +
		  2 * NSPACE * in.y +
		  NSPACE * NSPACE * in.z;
	}
}
site_idx calc_odd_spatial_idx(coord_full in)
{
	bool switcher = TMLQCD_CONV;
	if(switcher) {
		return
		  (uint)((in.x + in.w + 1) % 2) * (1 + 2 * in.z - (uint) (2 * in.z / NSPACE)) +
		  (uint)((in.w + in.x    ) % 2) * (    2 * in.z + (uint) (2 * in.z / NSPACE)) +
		  2 * NSPACE * in.y +
		  NSPACE * NSPACE * in.x;
	} else {
		return
		  (uint)((in.z + in.w + 1) % 2) * (1 + 2 * in.x - (uint) (2 * in.x / NSPACE)) +
		  (uint)((in.w + in.z    ) % 2) * (    2 * in.x + (uint) (2 * in.x / NSPACE)) +
		  2 * NSPACE * in.y +
		  NSPACE * NSPACE * in.z;
	}
}

/**
 * this takes a eo_site_idx (0..VOL4D/2) and returns its 4 coordinates
 * under the assumption that even-odd preconditioning is applied in the
 * x-y-plane as described above.
 * This is moved to the z-y plane if the tmlqcd conventions are used.
 * This is done for convenience since x and y direction have
 * the same extent.
 * Use the convention:
 *site_idx = x + y*NS + z*NS*NS + t*NS*NS*NS
 *= site_idx_spatial + t*VOLSPACE
 *and
 *spatial_idx = x * NS^XDIR-1 + y * NS^YDIR-1 + z * NS^ZDIR-1
 *= x + y * NS + z * NS^2
 *
 * Then one can "dissect" the site_idx i according to
 * t= i/VOLSPACE/2
 * z = (i-t*VOLSPACE/2)/(NS*NS/2)
 * y = (i-t*VOLSPACE/2 - z*NS*NS/2) / NS
 * x = (i-t*VOLSPACE/2 - z*NS*NS/2 - y*NS)
 * As mentioned above, y is taken to run from 0..NS/2
 */
coord_full dissect_eo_site_idx(const site_idx idx)
{
	coord_full tmp;
	bool switcher = TMLQCD_CONV;
	if(switcher) {
		tmp.z = idx;
		tmp.w = (int)(idx / (VOLSPACE / 2));
		tmp.z -= tmp.w * VOLSPACE / 2;
		tmp.x = (int)(tmp.z / (NSPACE * NSPACE / 2));
		tmp.z -= tmp.x * NSPACE * NSPACE / 2;
		tmp.y = (int)(tmp.z / NSPACE);
		tmp.z -= tmp.y * NSPACE;
	} else {
		tmp.x = idx;
		tmp.w = (int)(idx / (VOLSPACE / 2));
		tmp.x -= tmp.w * VOLSPACE / 2;
		tmp.z = (int)(tmp.x / (NSPACE * NSPACE / 2));
		tmp.x -= tmp.z * NSPACE * NSPACE / 2;
		tmp.y = (int)(tmp.x / NSPACE);
		tmp.x -= tmp.y * NSPACE;
	}
	return tmp;
}

/** given an eo site_idx (0..VOL4D/2), returns corresponding even site_idx */
st_idx get_even_st_idx(const site_idx idx)
{
	coord_full tmp = dissect_eo_site_idx(idx);
	st_idx res;
	res.space = calc_even_spatial_idx(tmp);
	res.time = tmp.w;
	return res;
}

/** given an eo site_idx (0..VOL4D/2), returns corresponding odd site_idx */
st_idx get_odd_st_idx(const site_idx idx)
{
	coord_full tmp = dissect_eo_site_idx(idx);
	st_idx res;
	res.space = calc_odd_spatial_idx(tmp);
	res.time = tmp.w;
	return res;
}

////////////////////////////////////////////////////////////////////
// Get positions of neighbours
// These functions do not rely explicitely on geometric convetions
////////////////////////////////////////////////////////////////////

/** returns neighbor in time direction given a temporal coordinate */
coord_temporal get_neighbor_temporal(const coord_temporal ntime)
{
	return (ntime + 1) % NTIME;
}

/** returns neighbor in time direction given a temporal coordinate */
coord_temporal get_lower_neighbor_temporal(const coord_temporal ntime)
{
	return (ntime - 1 + NTIME) % NTIME;
}

/** returns idx of neighbor in spatial direction dir given a spatial idx */
site_idx get_neighbor_spatial(const spatial_idx nspace, const dir_idx dir)
{
	coord_spatial coord = get_coord_spatial(nspace);
	switch(dir) {
		case XDIR:
			coord.x = (coord.x + 1) % NSPACE;
			break;
		case YDIR:
			coord.y = (coord.y + 1) % NSPACE;
			break;
		case ZDIR:
			coord.z = (coord.z + 1) % NSPACE;
			break;
	}
	return get_spatial_idx(coord);
}

/** returns idx of lower neighbor in spatial direction dir given a spatial idx */
site_idx get_lower_neighbor_spatial(const spatial_idx nspace, const dir_idx dir)
{
	coord_spatial coord = get_coord_spatial(nspace);
	switch(dir) {
		case XDIR:
			coord.x = (coord.x - 1 + NSPACE) % NSPACE;
			break;
		case YDIR:
			coord.y = (coord.y - 1 + NSPACE) % NSPACE;
			break;
		case ZDIR:
			coord.z = (coord.z - 1 + NSPACE) % NSPACE;
			break;
	}
	return get_spatial_idx(coord);
}

/** returns the st_idx of the neighbor in direction dir given a st_idx. */
st_idx get_neighbor_from_st_idx(const st_idx in, const dir_idx dir)
{
	st_idx tmp = in;
	if(dir == TDIR) tmp.time = get_neighbor_temporal(in.time);
	else tmp.space = get_neighbor_spatial(in.space, dir);
	return tmp;
}

/** returns the st_idx of the lower neighbor in direction dir given a st_idx. */
st_idx get_lower_neighbor_from_st_idx(const st_idx in, const dir_idx dir)
{
	st_idx tmp = in;
	if(dir == TDIR) tmp.time = get_lower_neighbor_temporal(in.time);
	else tmp.space = get_lower_neighbor_spatial(in.space, dir);
	return tmp;
}

// OPERATIONS ON CUSTOM DATA TYPES

//
//Functions implementing operations for the physical values
//

su3vec set_su3vec_zero()
{
	su3vec tmp;
	(tmp).e0 = hmc_complex_zero;
	(tmp).e1 = hmc_complex_zero;
	(tmp).e2 = hmc_complex_zero;
	return tmp;
}

su3vec su3vec_acc(su3vec in1, su3vec in2)
{
	su3vec tmp;
	tmp.e0.re = in1.e0.re + in2.e0.re;
	tmp.e0.im = in1.e0.im + in2.e0.im;
	tmp.e1.re = in1.e1.re + in2.e1.re;
	tmp.e1.im = in1.e1.im + in2.e1.im;
	tmp.e2.re = in1.e2.re + in2.e2.re;
	tmp.e2.im = in1.e2.im + in2.e2.im;
	return tmp;
}

su3vec su3vec_acc_i(su3vec in1, su3vec in2)
{
	su3vec tmp;
	tmp.e0.re = in1.e0.re - in2.e0.im;
	tmp.e0.im = in1.e0.im + in2.e0.re;
	tmp.e1.re = in1.e1.re - in2.e1.im;
	tmp.e1.im = in1.e1.im + in2.e1.re;
	tmp.e2.re = in1.e2.re - in2.e2.im;
	tmp.e2.im = in1.e2.im + in2.e2.re;
	return tmp;
}

su3vec su3vec_dim(su3vec in1, su3vec in2)
{
	su3vec tmp;
	tmp.e0.re = in1.e0.re - in2.e0.re;
	tmp.e0.im = in1.e0.im - in2.e0.im;
	tmp.e1.re = in1.e1.re - in2.e1.re;
	tmp.e1.im = in1.e1.im - in2.e1.im;
	tmp.e2.re = in1.e2.re - in2.e2.re;
	tmp.e2.im = in1.e2.im - in2.e2.im;
	return tmp;
}

su3vec su3vec_dim_i(su3vec in1, su3vec in2)
{
	su3vec tmp;
	tmp.e0.re = in1.e0.re + in2.e0.im;
	tmp.e0.im = in1.e0.im - in2.e0.re;
	tmp.e1.re = in1.e1.re + in2.e1.im;
	tmp.e1.im = in1.e1.im - in2.e1.re;
	tmp.e2.re = in1.e2.re + in2.e2.im;
	tmp.e2.im = in1.e2.im - in2.e2.re;
	return tmp;
}

su3vec su3vec_times_complex(const su3vec in, const hmc_complex factor)
{
	su3vec tmp;
	tmp.e0.re = in.e0.re * factor.re - in.e0.im * factor.im;
	tmp.e0.im = in.e0.im * factor.re + in.e0.re * factor.im;
	tmp.e1.re = in.e1.re * factor.re - in.e1.im * factor.im;
	tmp.e1.im = in.e1.im * factor.re + in.e1.re * factor.im;
	tmp.e2.re = in.e2.re * factor.re - in.e2.im * factor.im;
	tmp.e2.im = in.e2.im * factor.re + in.e2.re * factor.im;
	return tmp;
}

su3vec su3matrix_times_su3vec(Matrixsu3 u, su3vec in)
{
	su3vec tmp;

	tmp.e0.re = u.e00.re * in.e0.re + u.e01.re * in.e1.re + u.e02.re * in.e2.re
	            - u.e00.im * in.e0.im - u.e01.im * in.e1.im - u.e02.im * in.e2.im;
	tmp.e0.im = u.e00.re * in.e0.im + u.e01.re * in.e1.im + u.e02.re * in.e2.im
	            + u.e00.im * in.e0.re + u.e01.im * in.e1.re + u.e02.im * in.e2.re;

	tmp.e1.re = u.e10.re * in.e0.re + u.e11.re * in.e1.re + u.e12.re * in.e2.re
	            - u.e10.im * in.e0.im - u.e11.im * in.e1.im - u.e12.im * in.e2.im;
	tmp.e1.im = u.e10.re * in.e0.im + u.e11.re * in.e1.im + u.e12.re * in.e2.im
	            + u.e10.im * in.e0.re + u.e11.im * in.e1.re + u.e12.im * in.e2.re;

	tmp.e2.re = u.e20.re * in.e0.re + u.e21.re * in.e1.re + u.e22.re * in.e2.re
	            - u.e20.im * in.e0.im - u.e21.im * in.e1.im - u.e22.im * in.e2.im;
	tmp.e2.im = u.e20.re * in.e0.im + u.e21.re * in.e1.im + u.e22.re * in.e2.im
	            + u.e20.im * in.e0.re + u.e21.im * in.e1.re + u.e22.im * in.e2.re;

	return tmp;
}

su3vec su3matrix_dagger_times_su3vec(Matrixsu3 u, su3vec in)
{
	su3vec tmp;

	tmp.e0.re = u.e00.re * in.e0.re + u.e10.re * in.e1.re + u.e20.re * in.e2.re
	            + u.e00.im * in.e0.im + u.e10.im * in.e1.im + u.e20.im * in.e2.im;
	tmp.e0.im = u.e00.re * in.e0.im + u.e10.re * in.e1.im + u.e20.re * in.e2.im
	            - u.e00.im * in.e0.re - u.e10.im * in.e1.re - u.e20.im * in.e2.re;

	tmp.e1.re = u.e01.re * in.e0.re + u.e11.re * in.e1.re + u.e21.re * in.e2.re
	            + u.e01.im * in.e0.im + u.e11.im * in.e1.im + u.e21.im * in.e2.im;
	tmp.e1.im = u.e01.re * in.e0.im + u.e11.re * in.e1.im + u.e21.re * in.e2.im
	            - u.e01.im * in.e0.re - u.e11.im * in.e1.re - u.e21.im * in.e2.re;

	tmp.e2.re = u.e02.re * in.e0.re + u.e12.re * in.e1.re + u.e22.re * in.e2.re
	            + u.e02.im * in.e0.im + u.e12.im * in.e1.im + u.e22.im * in.e2.im;
	tmp.e2.im = u.e02.re * in.e0.im + u.e12.re * in.e1.im + u.e22.re * in.e2.im
	            - u.e02.im * in.e0.re - u.e12.im * in.e1.re - u.e22.im * in.e2.re;

	return tmp;
}
spinor set_spinor_zero()
{
	spinor tmp;
	tmp.e0 = set_su3vec_zero();
	tmp.e1 = set_su3vec_zero();
	tmp.e2 = set_su3vec_zero();
	tmp.e3 = set_su3vec_zero();
	return tmp;
}

spinor spinor_dim(spinor in1, spinor in2)
{
	spinor tmp;
	tmp.e0 = su3vec_dim(in1.e0, in2.e0);
	tmp.e1 = su3vec_dim(in1.e1, in2.e1);
	tmp.e2 = su3vec_dim(in1.e2, in2.e2);
	tmp.e3 = su3vec_dim(in1.e3, in2.e3);
	return tmp;
}

// END OF OPERATIONS ON CUSTOM DATA TYPES

// TODO document
Matrixsu3 getSU3SOA(__global const hmc_complex * const restrict in, const uint idx)
{
	return (Matrixsu3) {
		in[0 * GAUGEFIELD_STRIDE + idx],
		in[1 * GAUGEFIELD_STRIDE + idx],
		in[2 * GAUGEFIELD_STRIDE + idx],
		in[3 * GAUGEFIELD_STRIDE + idx],
		in[4 * GAUGEFIELD_STRIDE + idx],
		in[5 * GAUGEFIELD_STRIDE + idx],
		in[6 * GAUGEFIELD_STRIDE + idx],
		in[7 * GAUGEFIELD_STRIDE + idx],
		in[8 * GAUGEFIELD_STRIDE + idx]
	};
}

// TODO document
void putSU3SOA(__global hmc_complex * const restrict out, const uint idx, const Matrixsu3 val)
{
	out[0 * GAUGEFIELD_STRIDE + idx] = val.e00;
	out[1 * GAUGEFIELD_STRIDE + idx] = val.e01;
	out[2 * GAUGEFIELD_STRIDE + idx] = val.e02;
	out[3 * GAUGEFIELD_STRIDE + idx] = val.e10;
	out[4 * GAUGEFIELD_STRIDE + idx] = val.e11;
	out[5 * GAUGEFIELD_STRIDE + idx] = val.e12;
	out[6 * GAUGEFIELD_STRIDE + idx] = val.e20;
	out[7 * GAUGEFIELD_STRIDE + idx] = val.e21;
	out[8 * GAUGEFIELD_STRIDE + idx] = val.e22;
}

// TODO document
spinor getSpinorSOA_eo(__global const hmc_complex * const restrict in, const uint idx)
{
	return (spinor) {
		{
			// su3vec = 3 * cplx
			in[ 0 * EOPREC_SPINORFIELD_STRIDE + idx], in[ 1 * EOPREC_SPINORFIELD_STRIDE + idx], in[ 2 * EOPREC_SPINORFIELD_STRIDE + idx]
		}, {
			// su3vec = 3 * cplx
			in[ 3 * EOPREC_SPINORFIELD_STRIDE + idx], in[ 4 * EOPREC_SPINORFIELD_STRIDE + idx], in[ 5 * EOPREC_SPINORFIELD_STRIDE + idx]
		}, {
			// su3vec = 3 * cplx
			in[ 6 * EOPREC_SPINORFIELD_STRIDE + idx], in[ 7 * EOPREC_SPINORFIELD_STRIDE + idx], in[ 8 * EOPREC_SPINORFIELD_STRIDE + idx]
		}, {
			// su3vec = 3 * cplx
			in[ 9 * EOPREC_SPINORFIELD_STRIDE + idx], in[10 * EOPREC_SPINORFIELD_STRIDE + idx], in[11 * EOPREC_SPINORFIELD_STRIDE + idx]
		}
	};
}

// TODO document
void putSpinorSOA_eo(__global hmc_complex * const restrict out, const uint idx, const spinor val)
{
	// su3vec = 3 * cplx
	out[ 0 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e0.e0;
	out[ 1 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e0.e1;
	out[ 2 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e0.e2;

	// su3vec = 3 * cplx
	out[ 3 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e1.e0;
	out[ 4 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e1.e1;
	out[ 5 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e1.e2;

	// su3vec = 3 * cplx
	out[ 6 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e2.e0;
	out[ 7 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e2.e1;
	out[ 8 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e2.e2;

	// su3vec = 3 * cplx
	out[ 9 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e3.e0;
	out[10 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e3.e1;
	out[11 * EOPREC_SPINORFIELD_STRIDE + idx] = val.e3.e2;
}

/**
 @file fermionmatrix-functions for eoprec spinorfields
*/

//
// Variations of the part of the kernel working in one direction
//

//"local" dslash working on a particular link (n,t) of an eoprec field
//NOTE: each component is multiplied by +KAPPA, so the resulting spinor has to be mutliplied by -1 to obtain the correct dslash!!!
//the difference to the "normal" dslash is that the coordinates of the neighbors have to be transformed into an eoprec index

spinor dslash_eoprec_local(__global const hmc_complex * const restrict in, __global hmc_complex  const * const restrict field, const st_idx idx_arg, const dir_idx dir)
{
	//this is used to save the idx of the neighbors
	st_idx idx_tmp;

	spinor out_tmp, plus;
	site_idx nn_eo;
	su3vec psi, phi;
	Matrixsu3 U;
	//this is used to save the BC-conditions...
	hmc_complex bc_tmp;
	out_tmp = set_spinor_zero();

	//CP: all actions correspond to the mu = 0 ones

	///////////////////////////////////
	// mu = +1
	idx_tmp = get_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_arg));
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = KAPPA_SPATIAL_IM;
	/////////////////////////////////
	//Calculate (1 - gamma_1) y
	//with 1 - gamma_1:
	//| 1  0  0  i |       |       psi.e0 + i*psi.e3  |
	//| 0  1  i  0 | psi = |       psi.e1 + i*psi.e2  |
	//| 0  i  1  0 |       |(-i)*( psi.e1 + i*psi.e2) |
	//| i  0  0  1 |       |(-i)*( psi.e0 + i*psi.e3) |
	/////////////////////////////////
	psi = su3vec_acc_i(plus.e0, plus.e3);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e3 = su3vec_dim_i(out_tmp.e3, psi);

	psi = su3vec_acc_i(plus.e1, plus.e2);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e2 = su3vec_dim_i(out_tmp.e2, psi);

	///////////////////////////////////
	//mu = -1
	idx_tmp = get_lower_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_tmp));
	//in direction -mu, one has to take the complex-conjugated value of bc_tmp. this is done right here.
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = MKAPPA_SPATIAL_IM;
	///////////////////////////////////
	// Calculate (1 + gamma_1) y
	// with 1 + gamma_1:
	// | 1  0  0 -i |       |       psi.e0 - i*psi.e3  |
	// | 0  1 -i  0 | psi = |       psi.e1 - i*psi.e2  |
	// | 0  i  1  0 |       |(-i)*( psi.e1 - i*psi.e2) |
	// | i  0  0  1 |       |(-i)*( psi.e0 - i*psi.e3) |
	///////////////////////////////////
	psi = su3vec_dim_i(plus.e0, plus.e3);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e3 = su3vec_acc_i(out_tmp.e3, psi);

	psi = su3vec_dim_i(plus.e1, plus.e2);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e2 = su3vec_acc_i(out_tmp.e2, psi);

	return out_tmp;
}

void dslash_eoprec_local_noret(spinor * inout, __global const hmc_complex * const restrict in, __global hmc_complex  const * const restrict field, const st_idx idx_arg, const dir_idx dir)
{
	//this is used to save the idx of the neighbors
	st_idx idx_tmp;

	spinor plus;
	site_idx nn_eo;
	su3vec psi, phi;
	Matrixsu3 U;
	//this is used to save the BC-conditions...
	hmc_complex bc_tmp;

	//CP: all actions correspond to the mu = 0 ones

	///////////////////////////////////
	// mu = +1
	idx_tmp = get_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_arg));
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = KAPPA_SPATIAL_IM;
	/////////////////////////////////
	//Calculate (1 - gamma_1) y
	//with 1 - gamma_1:
	//| 1  0  0  i |       |       psi.e0 + i*psi.e3  |
	//| 0  1  i  0 | psi = |       psi.e1 + i*psi.e2  |
	//| 0  i  1  0 |       |(-i)*( psi.e1 + i*psi.e2) |
	//| i  0  0  1 |       |(-i)*( psi.e0 + i*psi.e3) |
	/////////////////////////////////
	psi = su3vec_acc_i(plus.e0, plus.e3);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	inout->e0 = su3vec_dim(inout->e0, psi);
	inout->e3 = su3vec_acc_i(inout->e3, psi);

	psi = su3vec_acc_i(plus.e1, plus.e2);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	inout->e1 = su3vec_dim(inout->e1, psi);
	inout->e2 = su3vec_acc_i(inout->e2, psi);

	///////////////////////////////////
	//mu = -1
	idx_tmp = get_lower_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_tmp));
	//in direction -mu, one has to take the complex-conjugated value of bc_tmp. this is done right here.
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = MKAPPA_SPATIAL_IM;
	///////////////////////////////////
	// Calculate (1 + gamma_1) y
	// with 1 + gamma_1:
	// | 1  0  0 -i |       |       psi.e0 - i*psi.e3  |
	// | 0  1 -i  0 | psi = |       psi.e1 - i*psi.e2  |
	// | 0  i  1  0 |       |(-i)*( psi.e1 - i*psi.e2) |
	// | i  0  0  1 |       |(-i)*( psi.e0 - i*psi.e3) |
	///////////////////////////////////
	psi = su3vec_dim_i(plus.e0, plus.e3);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	inout->e0 = su3vec_dim(inout->e0, psi);
	inout->e3 = su3vec_dim_i(inout->e3, psi);

	psi = su3vec_dim_i(plus.e1, plus.e2);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	inout->e1 = su3vec_dim(inout->e1, psi);
	inout->e2 = su3vec_dim_i(inout->e2, psi);
}

spinor dslash_eoprec_local_0(__global const hmc_complex * const restrict in, __global hmc_complex const * const restrict field, const st_idx idx_arg)
{
	spinor out_tmp, plus;
	//this is used to save the idx of the neighbors
	st_idx idx_tmp;
	dir_idx dir;
	site_idx nn_eo;
	su3vec psi, phi;
	Matrixsu3 U;
	//this is used to save the BC-conditions...
	hmc_complex bc_tmp;
	out_tmp = set_spinor_zero();

	//go through the different directions
	///////////////////////////////////
	// TDIR = 0, temporal
	///////////////////////////////////
	dir = TDIR;
	///////////////////////////////////
	//mu = +0
	idx_tmp = get_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_arg));
	//if chemical potential is activated, U has to be multiplied by appropiate factor
#ifdef _CP_REAL_
	U = multiply_matrixsu3_by_real (U, EXPCPR);
#endif
#ifdef _CP_IMAG_
	hmc_complex cpi_tmp = {COSCPI, SINCPI};
	U = multiply_matrixsu3_by_complex (U, cpi_tmp );
#endif
	bc_tmp.re = KAPPA_TEMPORAL_RE;
	bc_tmp.im = KAPPA_TEMPORAL_IM;
	///////////////////////////////////
	// Calculate psi/phi = (1 - gamma_0) plus/y
	// with 1 - gamma_0:
	// | 1  0  1  0 |        | psi.e0 + psi.e2 |
	// | 0  1  0  1 |  psi = | psi.e1 + psi.e3 |
	// | 1  0  1  0 |        | psi.e1 + psi.e3 |
	// | 0  1  0  1 |        | psi.e0 + psi.e2 |
	///////////////////////////////////
	// psi = 0. component of (1-gamma_0)y
	psi = su3vec_acc(plus.e0, plus.e2);
	// phi = U*psi
	phi =  su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e2 = su3vec_acc(out_tmp.e2, psi);
	// psi = 1. component of (1-gamma_0)y
	psi = su3vec_acc(plus.e1, plus.e3);
	// phi = U*psi
	phi =  su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e3 = su3vec_acc(out_tmp.e3, psi);

	/////////////////////////////////////
	//mu = -0
	idx_tmp = get_lower_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_tmp));
	//if chemical potential is activated, U has to be multiplied by appropiate factor
	//this is the same as at mu=0 in the imag. case, since U is taken to be U^+ later:
	//  (exp(iq)U)^+ = exp(-iq)U^+
	//as it should be
	//in the real case, one has to take exp(q) -> exp(-q)
#ifdef _CP_REAL_
	U = multiply_matrixsu3_by_real (U, MEXPCPR);
#endif
#ifdef _CP_IMAG_
	hmc_complex cpi_tmp = {COSCPI, SINCPI};
	U = multiply_matrixsu3_by_complex (U, cpi_tmp );
#endif
	//in direction -mu, one has to take the complex-conjugated value of bc_tmp. this is done right here.
	bc_tmp.re = KAPPA_TEMPORAL_RE;
	bc_tmp.im = MKAPPA_TEMPORAL_IM;
	///////////////////////////////////
	// Calculate psi/phi = (1 + gamma_0) y
	// with 1 + gamma_0:
	// | 1  0 -1  0 |       | psi.e0 - psi.e2 |
	// | 0  1  0 -1 | psi = | psi.e1 - psi.e3 |
	// |-1  0  1  0 |       | psi.e1 - psi.e2 |
	// | 0 -1  0  1 |       | psi.e0 - psi.e3 |
	///////////////////////////////////
	// psi = 0. component of (1+gamma_0)y
	psi = su3vec_dim(plus.e0, plus.e2);
	// phi = U*psi
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e2 = su3vec_dim(out_tmp.e2, psi);
	// psi = 1. component of (1+gamma_0)y
	psi = su3vec_dim(plus.e1, plus.e3);
	// phi = U*psi
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e3 = su3vec_dim(out_tmp.e3, psi);

	return out_tmp;
}

spinor dslash_eoprec_local_1(__global const hmc_complex * const restrict in, __global hmc_complex  const * const restrict field, const st_idx idx_arg)
{
	//this is used to save the idx of the neighbors
	st_idx idx_tmp;

	spinor out_tmp, plus;
	dir_idx dir;
	site_idx nn_eo;
	su3vec psi, phi;
	Matrixsu3 U;
	//this is used to save the BC-conditions...
	hmc_complex bc_tmp;
	out_tmp = set_spinor_zero();

	//CP: all actions correspond to the mu = 0 ones
	///////////////////////////////////
	// mu = 1
	///////////////////////////////////
	dir = XDIR;

	///////////////////////////////////
	// mu = +1
	idx_tmp = get_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_arg));
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = KAPPA_SPATIAL_IM;
	/////////////////////////////////
	//Calculate (1 - gamma_1) y
	//with 1 - gamma_1:
	//| 1  0  0  i |       |       psi.e0 + i*psi.e3  |
	//| 0  1  i  0 | psi = |       psi.e1 + i*psi.e2  |
	//| 0  i  1  0 |       |(-i)*( psi.e1 + i*psi.e2) |
	//| i  0  0  1 |       |(-i)*( psi.e0 + i*psi.e3) |
	/////////////////////////////////
	psi = su3vec_acc_i(plus.e0, plus.e3);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e3 = su3vec_dim_i(out_tmp.e3, psi);

	psi = su3vec_acc_i(plus.e1, plus.e2);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e2 = su3vec_dim_i(out_tmp.e2, psi);

	///////////////////////////////////
	//mu = -1
	idx_tmp = get_lower_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_tmp));
	//in direction -mu, one has to take the complex-conjugated value of bc_tmp. this is done right here.
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = MKAPPA_SPATIAL_IM;
	///////////////////////////////////
	// Calculate (1 + gamma_1) y
	// with 1 + gamma_1:
	// | 1  0  0 -i |       |       psi.e0 - i*psi.e3  |
	// | 0  1 -i  0 | psi = |       psi.e1 - i*psi.e2  |
	// | 0  i  1  0 |       |(-i)*( psi.e1 - i*psi.e2) |
	// | i  0  0  1 |       |(-i)*( psi.e0 - i*psi.e3) |
	///////////////////////////////////
	psi = su3vec_dim_i(plus.e0, plus.e3);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e3 = su3vec_acc_i(out_tmp.e3, psi);

	psi = su3vec_dim_i(plus.e1, plus.e2);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e2 = su3vec_acc_i(out_tmp.e2, psi);

	return out_tmp;
}

spinor dslash_eoprec_local_2(__global const hmc_complex * const restrict in, __global hmc_complex const * const restrict field, const st_idx idx_arg)
{
	//this is used to save the idx of the neighbors
	st_idx idx_tmp;

	spinor out_tmp, plus;
	dir_idx dir;
	site_idx nn_eo;
	su3vec psi, phi;
	Matrixsu3 U;
	//this is used to save the BC-conditions...
	hmc_complex bc_tmp;
	out_tmp = set_spinor_zero();

	///////////////////////////////////
	// mu = 2
	///////////////////////////////////
	dir = YDIR;

	///////////////////////////////////
	// mu = +2
	idx_tmp = get_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_arg));
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = KAPPA_SPATIAL_IM;
	///////////////////////////////////
	// Calculate (1 - gamma_2) y
	// with 1 - gamma_2:
	// | 1  0  0  1 |       |       psi.e0 + psi.e3  |
	// | 0  1 -1  0 | psi = |       psi.e1 - psi.e2  |
	// | 0 -1  1  0 |       |(-1)*( psi.e1 + psi.e2) |
	// | 1  0  0  1 |       |     ( psi.e0 + psi.e3) |
	///////////////////////////////////
	psi = su3vec_acc(plus.e0, plus.e3);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e3 = su3vec_acc(out_tmp.e3, psi);

	psi = su3vec_dim(plus.e1, plus.e2);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e2 = su3vec_dim(out_tmp.e2, psi);

	///////////////////////////////////
	//mu = -2
	idx_tmp = get_lower_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_tmp));
	//in direction -mu, one has to take the complex-conjugated value of bc_tmp. this is done right here.
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = MKAPPA_SPATIAL_IM;
	///////////////////////////////////
	// Calculate (1 + gamma_2) y
	// with 1 + gamma_2:
	// | 1  0  0 -1 |       |       psi.e0 - psi.e3  |
	// | 0  1  1  0 | psi = |       psi.e1 + psi.e2  |
	// | 0  1  1  0 |       |     ( psi.e1 + psi.e2) |
	// |-1  0  0  1 |       |(-1)*( psi.e0 - psi.e3) |
	///////////////////////////////////
	psi = su3vec_dim(plus.e0, plus.e3);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e3 = su3vec_dim(out_tmp.e3, psi);

	psi = su3vec_acc(plus.e1, plus.e2);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e2 = su3vec_acc(out_tmp.e2, psi);

	return out_tmp;
}

spinor dslash_eoprec_local_3(__global const hmc_complex * const restrict in, __global hmc_complex const * const restrict field, const st_idx idx_arg)
{
	//this is used to save the idx of the neighbors
	st_idx idx_tmp;

	spinor out_tmp, plus;
	dir_idx dir;
	site_idx nn_eo;
	su3vec psi, phi;
	Matrixsu3 U;
	//this is used to save the BC-conditions...
	hmc_complex bc_tmp;
	out_tmp = set_spinor_zero();

	///////////////////////////////////
	// mu = 3
	///////////////////////////////////
	dir = 3;

	///////////////////////////////////
	// mu = +3
	idx_tmp = get_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_arg));
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = KAPPA_SPATIAL_IM;
	///////////////////////////////////
	// Calculate (1 - gamma_3) y
	// with 1 - gamma_3:
	// | 1  0  i  0 |        |       psi.e0 + i*psi.e2  |
	// | 0  1  0 -i |  psi = |       psi.e1 - i*psi.e3  |
	// |-i  0  1  0 |        |   i *(psi.e0 + i*psi.e2) |
	// | 0  i  0  1 |        | (-i)*(psi.e1 - i*psi.e3) |
	///////////////////////////////////
	psi = su3vec_acc_i(plus.e0, plus.e2);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e2 = su3vec_dim_i(out_tmp.e2, psi);

	psi = su3vec_dim_i(plus.e1, plus.e3);
	phi = su3matrix_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e3 = su3vec_acc_i(out_tmp.e3, psi);

	///////////////////////////////////
	//mu = -3
	idx_tmp = get_lower_neighbor_from_st_idx(idx_arg, dir);
	//transform normal indices to eoprec index
	nn_eo = get_eo_site_idx_from_st_idx(idx_tmp);
	plus = getSpinorSOA_eo(in, nn_eo);
	U = getSU3SOA(field, get_link_idx_SOA(dir, idx_tmp));
	//in direction -mu, one has to take the complex-conjugated value of bc_tmp. this is done right here.
	bc_tmp.re = KAPPA_SPATIAL_RE;
	bc_tmp.im = MKAPPA_SPATIAL_IM;
	///////////////////////////////////
	// Calculate (1 + gamma_3) y
	// with 1 + gamma_3:
	// | 1  0 -i  0 |       |       psi.e0 - i*psi.e2  |
	// | 0  1  0  i | psi = |       psi.e1 + i*psi.e3  |
	// | i  0  1  0 |       | (-i)*(psi.e0 - i*psi.e2) |
	// | 0 -i  0  1 |       |   i *(psi.e1 + i*psi.e3) |
	///////////////////////////////////
	psi = su3vec_dim_i(plus.e0, plus.e2);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e0 = su3vec_acc(out_tmp.e0, psi);
	out_tmp.e2 = su3vec_acc_i(out_tmp.e2, psi);

	psi = su3vec_acc_i(plus.e1, plus.e3);
	phi = su3matrix_dagger_times_su3vec(U, psi);
	psi = su3vec_times_complex(phi, bc_tmp);
	out_tmp.e1 = su3vec_acc(out_tmp.e1, psi);
	out_tmp.e3 = su3vec_dim_i(out_tmp.e3, psi);

	return out_tmp;
}

//
// Kernels
//


__kernel void dslash_eoprec_1dir(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		out_tmp2 = dslash_eoprec_local_1(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__kernel void dslash_eoprec_2dirs(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		out_tmp2 = dslash_eoprec_local_1(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_2(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__kernel void dslash_eoprec_3dirs(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		out_tmp2 = dslash_eoprec_local_1(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_2(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_3(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__kernel void dslash_eoprec(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		out_tmp2 = dslash_eoprec_local_0(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_1(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_2(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_3(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__attribute__((reqd_work_group_size(128, 1, 1)))
__kernel void dslash_eoprec_lim_group_size(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		out_tmp2 = dslash_eoprec_local_0(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_1(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_2(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local_3(in, field, pos);
		out_tmp = spinor_dim(out_tmp, out_tmp2);

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__kernel void dslash_eoprec_simplified(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		out_tmp2 = dslash_eoprec_local(in, field, pos, TDIR);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local(in, field, pos, XDIR);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local(in, field, pos, YDIR);
		out_tmp = spinor_dim(out_tmp, out_tmp2);
		out_tmp2 = dslash_eoprec_local(in, field, pos, ZDIR);
		out_tmp = spinor_dim(out_tmp, out_tmp2);

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__kernel void dslash_eoprec_simplified_loop(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		for(int dir = 0; dir < NDIM; ++dir) {
			out_tmp2 = dslash_eoprec_local(in, field, pos, dir);
			out_tmp = spinor_dim(out_tmp, out_tmp2);
		}

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__kernel void dslash_eoprec_simplified_loop_unrolled(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		#pragma unroll NDIM
		for(int dir = 0; dir < NDIM; ++dir) {
			out_tmp2 = dslash_eoprec_local(in, field, pos, dir);
			out_tmp = spinor_dim(out_tmp, out_tmp2);
		}

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__kernel void dslash_eoprec_simplified_loop_nounroll(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();
		spinor out_tmp2;

		//calc dslash (this includes mutliplication with kappa)

		#pragma unroll 1
		for(int dir = 0; dir < NDIM; ++dir) {
			out_tmp2 = dslash_eoprec_local(in, field, pos, dir);
			out_tmp = spinor_dim(out_tmp, out_tmp2);
		}

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}

__kernel void dslash_eoprec_simplified_loop_noret(__global const hmc_complex * const restrict in, __global hmc_complex * const restrict out, __global const hmc_complex * const restrict field, const int evenodd)
{
	int global_size = get_global_size(0);
	int id = get_global_id(0);

	for(int id_tmp = id; id_tmp < EOPREC_SPINORFIELDSIZE; id_tmp += global_size) {
		st_idx pos = (evenodd == ODD) ? get_even_st_idx(id_tmp) : get_odd_st_idx(id_tmp);

		spinor out_tmp = set_spinor_zero();

		//calc dslash (this includes mutliplication with kappa)

		#pragma unroll 1
		for(int dir = 0; dir < NDIM; ++dir) {
			dslash_eoprec_local_noret(&out_tmp, in, field, pos, dir);
		}

		putSpinorSOA_eo(out, id_tmp, out_tmp);
	}
}


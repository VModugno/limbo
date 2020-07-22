//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|

#ifndef LIMBO_TOOLS_RANDOM_GENERATOR_HPP
#define LIMBO_TOOLS_RANDOM_GENERATOR_HPP

#include <Eigen/Core>

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <external/rand_utils.hpp>
#include <list>
#include <mutex>
#include <random>
#include <stdlib.h>
#include <utility>
#ifdef MATPLOTLIB
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif



namespace limbo {
    namespace tools {
        /// @ingroup tools
        /// a mt19937-based random generator (mutex-protected)
        ///
        /// usage :
        /// - RandomGenerator<dist<double>>(0.0, 1.0);
        /// - double r = rgen.rand();
        template <typename D>
        class RandomGenerator {
        public:
            using result_type = typename D::result_type;
            RandomGenerator(result_type a, result_type b,  int seed = -1) : _dist(a, b) { this->seed(seed); }

            result_type rand() { return _dist(_rgen); }

            void seed(int seed = -1)
            {
                if (seed >= 0)
                    _rgen.seed(seed);
                else
                    _rgen.seed(randutils::auto_seed_128{}.base());
            }

            void reset() { _dist.reset(); }

            void param(const typename D::param_type& param) { _dist.param(param); }

        private:
            D _dist;
            std::mt19937 _rgen;
        };
        // transform the bound from [0,1] to [-l,l]
		inline Eigen::VectorXd bound_transf(const Eigen::VectorXd& x, const Eigen::VectorXd& ub, const Eigen::VectorXd& lb){
			Eigen::VectorXd z(x.size());
			for(uint i = 0;i<x.size();i++){
				z[i] = x[i]*(ub[i]-lb[i]) + lb[i];
			}
			return z;
		}

		// transform the bound from [-l,l] to [0,1] z = (x-min)/ (max-min)
		inline Eigen::VectorXd bound_anti_transf(const Eigen::VectorXd& x, const Eigen::VectorXd& ub, const Eigen::VectorXd& lb){
			Eigen::VectorXd z(x.size());
			for(uint i = 0;i<x.size();i++){
				z[i] = (x[i]-lb[i])/(ub[i]-lb[i]);
			}
			return z;
		}

		//rotate the point
		inline Eigen::VectorXd rototrasl(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& R){
			Eigen::VectorXd z(x.size());
			z = mean + R*x;
			return z;
		}

		//rotate the point
		inline Eigen::VectorXd anti_rototrasl(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& R_inv){
			Eigen::VectorXd z(x.size());
			z = R_inv*(x-mean); //z=R^(-1)(x-mean);
			return z;
		}

        // create meshgrid on 2d plane
        inline std::vector<Eigen::MatrixXd> meshgrid_2d(double step_x,double low_x,double high_x,double step_y,double low_y,double high_y){

        	std::vector<Eigen::MatrixXd> res(2);
        	int size_x = round((high_x - low_x)/step_x);
        	int size_y = round((high_y - low_y)/step_y);
        	Eigen::VectorXd x_coord = Eigen::VectorXd::LinSpaced(size_x,low_x,high_x);
        	Eigen::VectorXd y_coord = Eigen::VectorXd::LinSpaced(size_y,low_y,high_y);

        	Eigen::MatrixXd _x(x_coord.size(),y_coord.size());
        	Eigen::MatrixXd _y(x_coord.size(),y_coord.size());

        	for(unsigned int i = 0;i<y_coord.size();i++){
        		for(unsigned int j = 0;j<x_coord.size();j++){
        			//Eigen::VectorXd cur(2);
        			_x(i,j) = x_coord[j];
        			_y(i,j) = y_coord[i];
        		    //res.push_back(cur);
        		}
        	}
        	 res[0]=_x;
        	 res[1]=_y;
        	 return res;
        }

        inline std::vector<Eigen::MatrixXd> rotated_meshgrid_2d(const Eigen::VectorXd& mean, const Eigen::MatrixXd& R,double size_x,double low_x,double high_x,double size_y,double low_y,double high_y){
        	std::vector<Eigen::MatrixXd> res;
        	res = meshgrid_2d(size_x,low_x,high_x,size_y,low_y,high_y);
        	for(unsigned int i = 0;i<res[0].rows();i++){
        		for(unsigned int j = 0;j<res[0].cols();j++){
        			Eigen::VectorXd cur(2);
        			cur[0] = res[0](i,j);
        			cur[1] = res[1](i,j);
        			cur = rototrasl(cur,mean,R);
        			res[0](i,j) = cur[0];
        			res[1](i,j) = cur[1];
        		}
        	}
        	return res;
        }

        inline std::vector<Eigen::MatrixXd> scale_rot_meshgrid_2d(const Eigen::VectorXd& mean, const Eigen::MatrixXd& R,const Eigen::VectorXd& ub, const Eigen::VectorXd& lb,double size_x,double low_x,int high_x,double size_y,double low_y,double high_y){
			std::vector<Eigen::MatrixXd> res;
			res = meshgrid_2d(size_x,low_x,high_x,size_y,low_y,high_y);

			for(unsigned int i = 0;i<res[0].rows();i++){
				for(unsigned int j = 0;j<res[0].cols();j++){
					Eigen::VectorXd cur(2);
					cur[0] = res[0](i,j);
					cur[1] = res[1](i,j);
					cur = bound_transf(cur,ub,lb);
					cur = rototrasl(cur,mean,R);
					res[0](i,j) = cur[0];
					res[1](i,j) = cur[1];
				}
			}
			return res;
		}

		#ifdef MATPLOTLIB
		// drawing function
		inline void plot_point(const Eigen::VectorXd& x, int dot_size){
			std::vector<double> x_dot (1), y_dot(1);
			x_dot.at(0) = x[0];
		    y_dot.at(0) = x[1];
		    plt::scatter(x_dot, y_dot,dot_size);
		}
		// drawing function
		inline void plot_points(const std::vector<Eigen::VectorXd>& x, int dot_size){
			for(uint i=0;i<x.size();i++){
				plot_point(x[i],dot_size);
			}
		}

		// starting from the south west point going counterclockwise
		inline void plot_rotated_box(const Eigen::MatrixXd& R,const Eigen::VectorXd& center, double width,double height){
			Eigen::VectorXd sw(2);
			Eigen::VectorXd se(2);
			Eigen::VectorXd ne(2);
			Eigen::VectorXd nw(2);
			std::vector<double> x_coord (5), y_coord(5);
			//south-west
			sw[0] = -width/2;
			sw[1] = -height/2;
			//south-east
			se[0] = +width/2;
			se[1] = -height/2;
			//north-east
			ne[0] = +width/2;
			ne[1] = +height/2;
			//north-west
			nw[0] = -width/2;
			nw[1] = +height/2;

			// rotate point in the original frame
			sw = center + R*sw;
			se = center + R*se;
			ne = center + R*ne;
			nw = center + R*nw;

			x_coord.at(0) = sw[0];
			y_coord.at(0) = sw[1];
		    x_coord.at(1) = se[0];
		    y_coord.at(1) = se[1];
		    x_coord.at(2) = ne[0];
		    y_coord.at(2) = ne[1];
		    x_coord.at(3) = nw[0];
		  	y_coord.at(3) = nw[1];
		  	// i need to repeat the first point to close the figure
		  	x_coord.at(4) = sw[0];
		    y_coord.at(4) = sw[1];

		  	plt::plot(x_coord, y_coord);
		}

		inline void lim_img(int xmin, int xmax, int ymin, int ymax){
			plt::xlim(xmin,xmax);    // x-axis interval: [0, 1e6]
			plt::ylim(ymin,ymax);
		}

		inline void show_img(){
			plt::show();
		}
		#endif

        /// @ingroup tools
        using rdist_double_t = std::uniform_real_distribution<double>;
        /// @ingroup tools
        using rdist_int_t = std::uniform_int_distribution<int>;
        /// @ingroup tools
        using rdist_gauss_t = std::normal_distribution<>;

        /// @ingroup tools
        /// Double random number generator
        using rgen_double_t = RandomGenerator<rdist_double_t>;

        /// @ingroup tools
        /// Double random number generator (gaussian)
        using rgen_gauss_t = RandomGenerator<rdist_gauss_t>;

        /// @ingroup tools
        /// integer random number generator
        using rgen_int_t = RandomGenerator<rdist_int_t>;

        /// @ingroup tools
        /// random vector by providing custom RandomGenerator
        template <typename Rng>
        inline Eigen::VectorXd random_vec(int size, Rng& rng)
        {
            Eigen::VectorXd res(size);
            for (int i = 0; i < size; ++i)
                res[i] = rng.rand();
            return res;
        }

        /// @ingroup tools
        /// random vector in [0, 1]
        ///
        /// - this function is thread safe because we use a random generator for each thread
        /// - we use a C++11 random number generator
        inline Eigen::VectorXd random_vector_bounded(int size)
        {
            static thread_local rgen_double_t rgen(0.0, 1.0);
            return random_vec(size, rgen);
        }

        /// @ingroup tools
        /// random vector generated with a normal distribution centered on 0, with standard deviation of 10
        ///
        /// - this function is thread safe because we use a random generator for each thread
        /// - we use a C++11 random number generator
        inline Eigen::VectorXd random_vector_unbounded(int size)
        {
            static thread_local rgen_gauss_t rgen(0.0, 10.0);
            return random_vec(size, rgen);
        }

        /// @ingroup tools
        /// random vector wrapper for both bounded and unbounded versions
        inline Eigen::VectorXd random_vector(int size, bool bounded = true)
        {
            if (bounded)
                return random_vector_bounded(size);
            return random_vector_unbounded(size);
        }

        /// random vector wrapper for both bounded and unbounded versions
		inline Eigen::VectorXd random_vector(int size, const Eigen::VectorXd& ub,const Eigen::VectorXd& lb)
		{

			Eigen::VectorXd z = random_vector_bounded(size);
			return bound_transf(z,ub,lb);

		}

        /// @ingroup tools
        /// generate n random samples with Latin Hypercube Sampling (LHS) in [0, 1]^dim
        inline Eigen::MatrixXd random_lhs(int dim, int n)
        {
            Eigen::VectorXd cut = Eigen::VectorXd::LinSpaced(n + 1, 0., 1.);
            Eigen::MatrixXd u = Eigen::MatrixXd::Zero(n, dim);

            for (int i = 0; i < n; i++) {
                u.row(i) = tools::random_vector(dim, true);
            }

            Eigen::VectorXd a = cut.head(n);
            Eigen::VectorXd b = cut.tail(n);

            Eigen::MatrixXd rdpoints = Eigen::MatrixXd::Zero(n, dim);
            for (int i = 0; i < dim; i++) {
                rdpoints.col(i) = u.col(i).array() * (b - a).array() + a.array();
            }

            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, dim);
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(n);
            static thread_local std::mt19937 rgen(randutils::auto_seed_128{}.base());
            for (int i = 0; i < dim; i++) {
                perm.setIdentity();
                std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), rgen);
                Eigen::MatrixXd tmp = perm * rdpoints;
                H.col(i) = tmp.col(i);
            }

            return H;
        }

    } // namespace tools
} // namespace limbo

#endif

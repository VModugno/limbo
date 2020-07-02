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
#ifndef LIMBO_ACQUI_ECI_HPP
#define LIMBO_ACQUI_ECI_HPP

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <Eigen/Core>
#include <vector>
#include <math.h>

// Careful! this version of eci works only with local_onestep_boptimizer

namespace limbo {
    namespace defaults {
        struct acqui_eci {
            /// @ingroup acqui_defaults
            BO_PARAM(double, xi, 0.9);
        };
    }
   // m = mean
   // s = variance
   inline double my_cdf(double m, double s,double x){
	   double cdf = 0.5 * std::erfc( -(x-m)/(s*sqrt(2)) );
	   return cdf;
    }
    inline double my_pdf(double m, double s, double x){
	   double pdf = std::exp( -std::pow(x-m,2)/(2*std::pow(s,2)) )/(s*sqrt(2*M_PI));
	   return pdf;
    }
    namespace acqui {
        /** @ingroup acqui
        \rst
        GP-UCB (Upper Confidence Bound). See :cite:`brochu2010tutorial`, p. 15. See also: http://arxiv.org/abs/0912.3995

        .. math::
          UCB(x) = \mu(x) + \kappa \sigma(x).

        with:

        .. math::
          \kappa = \sqrt{2 \log{(\frac{n^{D/2+2}\pi^2}{3 \delta})}}

        where :math:`n` is the number of past evaluations of the objective function and :math:`D` the dimensionality of the parameters (dim_in).

        Parameters:
          - `double delta` (a small number in [0,1], e.g. 0.1)
        \endrst
        */
        template <typename Params,typename Model >
        class ECI {
        public:

        	//ECI(const Model & model, int iteration): _model(model)
        	//{
        	//    double nt = std::pow(iteration, dim_in() / 2.0 + 2.0);
        	//    _xi = Params::acqui_eci::xi();
            //    _constrained = Params::bayes_opt_boptimizer::constrained();
            //}
           ECI(){};
           ECI(const Model& model,const std::vector<Model>&  model_constr,double fmax, int iteration = 0): _model(model), _models_constr(model_constr),_nb_samples(-1)
           {
                _xi = Params::acqui_eci::xi();
                _constrained = Params::bayes_opt_bobase::constrained();
                _f_max = fmax;
           }

            size_t dim_in() const { return _model.dim_in(); }

            size_t dim_out() const { return _model.dim_out(); }


            template <typename AggregatorFunction>
            opt::eval_t operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun, bool gradient)
            {
            	// DEBUG
            	//std::cout << "sono dentro eci!" << std::endl;
                assert(!gradient);
                Eigen::VectorXd mu;
                double sigma_sq,z,ret1,ret2;
                std::tie(mu, sigma_sq) = _model.query(v);
                double sigma = std::sqrt(sigma_sq);

                //std::cout << "model Prediction = " << afun(mu) << std::endl;
                // If \sigma(x) = 0 or we do not have any observation yet we return 0
				if (sigma < 1e-10 || _model.samples().size() < 1)
					return opt::no_grad(0.0);

				if (_nb_samples != _model.nb_samples()) {
					std::vector<double> rewards;
					for (auto s : _model.samples()) {
						// DEBUG
						//std::cout << "reward = " << afun(_model.mu(s)) << std::endl; ;
						rewards.push_back(afun(_model.mu(s)));
					}

					_nb_samples = _model.nb_samples();
					_f_max = *std::max_element(rewards.begin(), rewards.end());
					// DEBUG
					//std::cout << "_f_max = "<< _f_max <<std::endl;
				}


			    z = (afun(mu) - _f_max - _xi)/sigma;
			    //DEBUG
			    //std::cout << "z = " << z << std::endl;
                ret1 = (afun(mu) - _f_max -_xi) * my_cdf(0,1,z) + sigma*my_pdf(0,1,z);



                ret2 = 1.0;
                if(_constrained){
					for(uint i=0;i<_models_constr.size();i++){
						std::tie(mu, sigma_sq) = _models_constr[i].query(v);
						sigma = std::sqrt(sigma_sq);
						// DEBUG
						//if(afun(mu)<0)
						std::cout << "constr["<<i<<"]  mean = "<<afun(mu)<< " var_sq = "<<sigma_sq<<" var =" << sigma<< std::endl;
						ret2 = ret2* my_cdf(afun(mu),sigma,0);
					}
                }
                //DEBUG
                //std::cout <<"input = " << v.transpose() << std::endl;
                std::cout << "ret1 = "<< ret1<< " ret2 = "<< ret2 << std::endl;
                return opt::no_grad(ret2*ret1);
            }

        protected:
            const Model& _model;
            const std::vector<Model>& _models_constr;
            double _xi;
            bool _constrained;
            int _nb_samples;
            double _f_max;
        };
    }
}
#endif

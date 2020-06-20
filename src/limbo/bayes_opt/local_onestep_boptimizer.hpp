/*
 * local_onestep_boptimizer.hpp
 *
 *  Created on: May 15, 2020
 *      Author: vale
 */

// the hypothesis here is that the particles does not speak with each other


#ifndef SRC_LIMBO_BAYES_OPT_LOCAL_ONESTEP_BOPTIMIZER_HPP_
#define SRC_LIMBO_BAYES_OPT_LOCAL_ONESTEP_BOPTIMIZER_HPP_

// WARNING: local one step optimizer work only with acqui_manager as acquisition method!


#include <algorithm>
#include <iostream>
#include <iterator>
#include <cmath>

#include <boost/parameter/aux_/void.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <limbo/bayes_opt/bo_base.hpp>
#include <limbo/acqui/acqui_manager.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>
#ifdef USE_NLOPT
#include <limbo/opt/nlopt_no_grad.hpp>
#elif defined USE_LIBCMAES
#include <limbo/opt/cmaes.hpp>
#else
#include <limbo/opt/grid_search.hpp>
#endif



namespace limbo {
    namespace defaults {
        struct local_bayes_opt_onestep_boptimizer {
            BO_PARAM(int, hp_period, -1);
            BO_PARAM(double,sigma_multiplier,1); // this parameters regulate the bound
            BO_PARAM(double, thresh, 0.1);
        };
    }
    //TODO particle data (mean, covariance and sigma have to be adapted to account for the fact
    // our parameter space is in [0,1])

    // needed to manage data from particle
    struct ParticleData{
    	double          _k;
    	double          _sigma;
    	Eigen::VectorXd _mean;
    	Eigen::MatrixXd _cov;
    	Eigen::MatrixXd _rot;
    	Eigen::VectorXd _zoom_bound;

    	ParticleData(){};

    	ParticleData(int size){
    		_sigma      = 0;
    		_mean       = Eigen::VectorXd::Zero(size);
    		_cov        = Eigen::MatrixXd::Zero(size,size);
    		_zoom_bound = Eigen::VectorXd::Zero(size);
			_rot        = Eigen::MatrixXd::Zero(size,size);
			boost::math::chi_squared mydist(size);
			_k          = quantile(mydist, 0.95);


		};

    	ParticleData(double sigma,const Eigen::VectorXd& mean,const Eigen::MatrixXd& cov):_sigma(sigma),_mean(mean),_cov(cov){
    		_zoom_bound = Eigen::VectorXd::Zero(cov.rows());
    		_rot         = Eigen::MatrixXd::Zero(cov.rows(),cov.cols());
    		boost::math::chi_squared mydist(cov.rows());
    		_k           = quantile(mydist, 0.95);


    	};

    	inline bool operator==(const ParticleData& rhs){
    		if(this->_mean == rhs._mean)
    			return true;
    		else
    			return false;
    	};
    	void update(const ParticleData& d){
    		_k     = d._k;
    		_mean  = d._mean;
    		_cov   = d._cov;
    		_sigma = d._sigma;

    	}
    	void compute_bound_and_rot(){
    		Eigen::EigenSolver<Eigen::MatrixXd> eig(_cov);
    		_rot        = eig.eigenvectors().real();
    		_zoom_bound = _k * ((eig.eigenvalues()).cwiseAbs()).cwiseSqrt() *_sigma;
    		//DEBUG
            #ifdef PLOT_BO
    		std::cout << "_rot = " << _rot<<std::endl;
    		std::cout << "_zoom_bound = " << _zoom_bound.transpose()<<std::endl;
			#endif
    	}
    };


    namespace bayes_opt {

        using local_onestep_boptimizer_signature = boost::parameter::parameters<boost::parameter::optional<tag::acquiopt>,
            boost::parameter::optional<tag::statsfun>,
            boost::parameter::optional<tag::initfun>,
            boost::parameter::optional<tag::acquifun>,
            boost::parameter::optional<tag::stopcrit>,
            boost::parameter::optional<tag::modelfun>>;

        // clang-format off
        /**
        The classic Bayesian optimization algorithm.

        \rst
        References: :cite:`brochu2010tutorial,Mockus2013`
        \endrst

        This class takes the same template parameters as BoBase. It adds:
        \rst
        +---------------------+------------+----------+---------------+
        |type                 |typedef     | argument | default       |
        +=====================+============+==========+===============+
        |acqui. optimizer     |acquiopt_t  | acquiopt | see below     |
        +---------------------+------------+----------+---------------+
        \endrst

        The default value of acqui_opt_t is:
        - ``opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>`` if NLOpt was found in `waf configure`
        - ``opt::Cmaes<Params>`` if libcmaes was found but NLOpt was not found
        - ``opt::GridSearch<Params>`` otherwise (please do not use this: the algorithm will not work as expected!)
        */
        template <class Params,
          class A1 = boost::parameter::void_,
          class A2 = boost::parameter::void_,
          class A3 = boost::parameter::void_,
          class A4 = boost::parameter::void_,
          class A5 = boost::parameter::void_,
          class A6 = boost::parameter::void_>
        // clang-format on
        class LocalOneStepBOptimizer : public BoBase<Params, A1, A2, A3, A4, A5, A6> {
        public:
            // defaults
            struct defaults {
#ifdef USE_NLOPT
                using acquiopt_t = opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>;
#elif defined(USE_LIBCMAES)
                using acquiopt_t = opt::Cmaes<Params>;
#else
#warning NO NLOpt, and NO Libcmaes: the acquisition function will be optimized by a grid search algorithm (which is usually bad). Please install at least NLOpt or libcmaes to use limbo!.
                using acquiopt_t = opt::GridSearch<Params>;
#endif
                //using acqui_t = acqui::AcquiManager<Params, model::GP<Params> >;
            };
            /// link to the corresponding BoBase (useful for typedefs)
            using base_t = BoBase<Params, A1, A2, A3, A4, A5, A6>;
            using model_t = typename base_t::model_t;
            using acquisition_function_t = typename base_t::acquisition_function_t;
            // extract the types
            using args = typename local_onestep_boptimizer_signature::bind<A1, A2, A3, A4, A5, A6>::type;
            //using acquisition_function_t = typename boost::parameter::binding<args, tag::acquifun, typename defaults::acqui_t>::type;
            using acqui_optimizer_t = typename boost::parameter::binding<args, tag::acquiopt, typename defaults::acquiopt_t>::type;

            // main functions-----------------------------------------------------------------------------------------
            //we need to initialize the empty vector of gp model just once at the beginning (call it once)
            // the init_list should contain point that are contained in the particle covariance
            // the hyp is that I receive a sample in the original dimension and i transform to the [0,1] bound
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void init(const StateFunction& sfun, const ParticleData& d, const std::vector<Eigen::VectorXd>& init_list, const Eigen::VectorXd& UB, const Eigen::VectorXd& LB, const AggregatorFunction& afun = AggregatorFunction()){

            	_constrained = Params::bayes_opt_bobase::constrained();
            	_d.update(d);
            	_d.compute_bound_and_rot();
            	_dim_in         = StateFunction::dim_in();
            	_dim_out        = StateFunction::dim_out();
            	_constr_dim_out = StateFunction::constr_dim_out();
            	_ub             = UB; // not used for now
            	_lb             = LB; // not used for now
            	// initialize dimension inside the bo_base class
            	this->simple_init(sfun);
            	// provide a list of sample (one or more)
            	for(uint i =0;i<init_list.size();i++){
            		// convert from [-l,l] to [0,1] // the hyp is that i get this
            		// sample from another module and they are not generated by the
            		// sampler from limbo as in the case of the legacy optimizer
					// update data in bo_base

            		// TOCHECK, maybe here we can avoid transform back and forth the sample
            		//Eigen::VectorXd sample = bound_anti_transf(init_list[i], _ub, _lb);
					this->eval_and_add(sfun,init_list[i]);
				}
            	// here i rebuild the model
            	double dist = _d._zoom_bound.maxCoeff();
                init_local_model(dist);
            }
            // the hyp is that I receive a sample in the original dimension and i transform to the [0,1] bound
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void init(const StateFunction& sfun, const ParticleData& d, const std::vector<Eigen::VectorXd>& init_list, const std::vector<Eigen::VectorXd>& obs, const Eigen::VectorXd& UB, const Eigen::VectorXd& LB, const AggregatorFunction& afun = AggregatorFunction()){

				_constrained = Params::bayes_opt_bobase::constrained();
				_d.update(d);
				_d.compute_bound_and_rot();
				_dim_in         = StateFunction::dim_in();
				_dim_out        = StateFunction::dim_out();
				_constr_dim_out = StateFunction::constr_dim_out();
				_ub             = UB;
				_lb             = LB;
				// initialize dimension inside the bo_base class
				this->simple_init(sfun);
				// provide a list of sample (one or more)
				for(uint i =0;i<init_list.size();i++){
					// convert from [-l,l] to [0,1] // the hyp is that i get this
					// sample from another module and they are not generated by the
					// sampler from limbo as in the case of the legacy optimizer
					// update data in bo_base

					// TOCHECK, maybe here we can avoid transform back and forth the sample
					//Eigen::VectorXd sample = bound_anti_transf(init_list[i], _ub, _lb);
					this->add_new_sample(init_list[i],obs[i]);
				}
				// here i rebuild the model
				double dist = _d._zoom_bound.maxCoeff();
				init_local_model(dist);
			}

            // we need to call it at every new sample from (1+1-cmaes)
            template <typename AggregatorFunction = FirstElem>
		    void update_bo(const Eigen::VectorXd& sample,const Eigen::VectorXd& sol, const ParticleData& d, const AggregatorFunction& afun = AggregatorFunction()){

            	// we need to transform back the sample from its bound to zero one range for each dimension


            	if(_d == d){ // we need to add the current point to the one we are working on (the particle did not  move)
					// TOCHECK
					//update_data(bound_anti_transf(sample,_ub, _lb),sol);
            		// even if the mean did not change i update the particle data to get the latest covariance matrix
            	    _d.update(d);
					update_data(sample,sol);
            	}
            	else{  // we need to create a new local gp model (the particle is moving so we need to ditch the old gp model and create a new one)
            		//update particle data
					_d.update(d);
					// update rotation matrix and bounds (i have to do this here in order to ge the point close to the current meand and build the new local GP)
					_d.compute_bound_and_rot();
					// add current sample to bo_base
					//this->add_new_sample(bound_anti_transf(sample,_ub, _lb),sol);
					// TOCHECK
					this->add_new_sample(sample,sol);
					// check if i can reuse some of the older points
					double dist = _d._zoom_bound.maxCoeff();
					// here i rebuild the model
					init_local_model(dist);
					// update stats
					this->_update_stats(*this, afun);

					this->_current_iteration++;
					this->_total_iterations++;
            	}
		    }

            // The main function (run the Bayesian optimization algorithm) (call it when model is sufficiently good to be used)
            template <typename AggregatorFunction = FirstElem>
            Eigen::VectorXd optimize(std::string strategy,const AggregatorFunction& afun = AggregatorFunction())
            {
                acqui_optimizer_t acqui_optimizer;

                // update hyperparameters
				if (Params::local_bayes_opt_onestep_boptimizer::hp_period() > 0
					&& (this->_current_iteration + 1) % Params::local_bayes_opt_onestep_boptimizer::hp_period() == 0){
					//DEBUG
					#ifdef PLOT_BO
					std::cout << "updating kernel parameters" << std::endl;
                    #endif
					for(uint i = 0;i<this->_observations.size();i++){
						if(i == 0)
							_model.optimize_hyperparams();
						else if(_constrained)
							_models_constr[i-1].optimize_hyperparams();
					}
				}

				// update rotation matrix and bound for the zooming step (optimization)
				_d.compute_bound_and_rot();
				// optimization step
				acquisition_function_t acqui(_model,_models_constr, strategy, this->_current_iteration);
				auto acqui_optimization = [&](const Eigen::VectorXd& x, bool g) { return acqui(tools::rototrasl(tools::bound_transf(x,_d._zoom_bound,-_d._zoom_bound),_d._mean,_d._rot), afun, g); };
				// here i select point in the [0,1] than i transform them in the covariance space and finally i transform them back in the original space
				Eigen::VectorXd starting_point = tools::random_vector(_dim_in, Params::bayes_opt_bobase::bounded());
				// DEBUG
				Eigen::VectorXd a = tools::bound_transf(starting_point,_d._zoom_bound,-_d._zoom_bound);
				Eigen::VectorXd b = tools::rototrasl(a,_d._mean,_d._rot);

				Eigen::VectorXd max_sample = acqui_optimizer(acqui_optimization, starting_point, Params::bayes_opt_bobase::bounded());



				// bring to the original space the max sample (it is in zero one at this point)
				max_sample = tools::bound_transf(max_sample,_d._zoom_bound,-_d._zoom_bound);
				max_sample = tools::rototrasl(max_sample,_d._mean,_d._rot);


				return max_sample;

            }

            // for debug purpose only
            template <typename StateFunction>
            Eigen::VectorXd eval(Eigen::VectorXd& sample,const StateFunction& sfun){
            	// we need to transform back the sample from zero one range to its original bound
            	// TOCHECK
            	//sample = bound_transf(sample,_ub, _lb);

            	Eigen::VectorXd	sol = Eigen::VectorXd(sfun(sample));
            	return sol;
            }


            // auxiliary functions------------------------------------------------------------------------------------
            template <typename AggregatorFunction = FirstElem>
            void update_data(const Eigen::VectorXd& sample,const Eigen::VectorXd& sol,const AggregatorFunction& afun = AggregatorFunction()){
            	// update data in bo_base
            	this->add_new_sample(sample,sol);
                // update model
				for(uint i = 0;i<this->_observations.size();i++){
					if(i == 0)
						_model.add_sample(this->_samples.back(), this->_observations[i].back());
					else if(_constrained)
						_models_constr[i-1].add_sample(this->_samples.back(), this->_observations[i].back());
				}
				// VALE update stats
				this->_update_stats(*this, afun);

				this->_current_iteration++;
			    this->_total_iterations++;

            }

            // here i want to do side effect on  the last two inputs
            void init_local_model(double dist){
            	std::vector<Eigen::VectorXd>  local_sample;
            	std::vector< std::vector<Eigen::VectorXd>> local_obs;

            	// we destroy the former models and build new ones (this work even the first time for the initialization)
				_model = model_t(_dim_in,_dim_out);
				if(_constrained){
					_models_constr.clear();
					for(int i = 0;i<_constr_dim_out;i++)
						_models_constr.push_back( model_t(_dim_in,1) );
				}

            	// initialize local observations
            	if(_constrained){
					for(int i = 0; i < _constr_dim_out + _dim_out;i++){
						std::vector<Eigen::VectorXd> cur;
						local_obs.push_back(cur);
					}
            	}else{
            		for(int i = 0; i < _dim_out;i++){
						std::vector<Eigen::VectorXd> cur;
						local_obs.push_back(cur);
					}

            	}

            	// find samples close to mean
            	for(uint i =0;i<this->_samples.size();i++){
            		double diff =  (this->_samples[i] - _d._mean).norm();
            		if(diff <= dist){
            			local_sample.push_back(this->_samples[i]);
            			for(uint j = 0;j< this->_observations.size();j++){
            				local_obs[j].push_back(this->_observations[j][i]);
            			}
            		}
            	}
            	// compute gp models (if i have no local sample i rise an error)
            	if(local_sample.size()>0){
					for(uint i = 0;i<this->_observations.size();i++){
						if(i==0)
							_model.compute(local_sample, local_obs[i]);
						else if(Params::bayes_opt_bobase::constrained()){
							_models_constr[i-1].compute(local_sample, local_obs[i]);
						}
					}
            	}else{
            		std::cout<< "local sample cannot be empty and it needs to be fixed!"<< std::endl;
            	}
            }

            template <typename AggregatorFunction = FirstElem>
            bool evaluate_local_model(const Eigen::VectorXd& sample,const Eigen::VectorXd& real_mu,const AggregatorFunction& afun = AggregatorFunction()){
            	Eigen::VectorXd mu;
				double sigma_sq;

				//TESTING update parameters for the model
				_model.optimize_hyperparams();
				// we need to transform back the sample from its bound to zero one range for each dimension
				// TOCHECK
				//std::tie(mu, sigma_sq) = _model.query(bound_anti_transf(sample,_ub, _lb));
				std::tie(mu, sigma_sq) = _model.query(sample);
				double sigma  = std::sqrt(sigma_sq);
				double diff   = abs(afun(mu) - afun(real_mu));
				// DEBUG
			    std::cout<<"surrogate model error = "<<diff<<std::endl;
				double thresh = Params::local_bayes_opt_onestep_boptimizer::thresh();
				if( diff < thresh)
					return true;
				else
					return false;
            }

            /// return the best observation so far (i.e. max(f(x)))
			template <typename AggregatorFunction = FirstElem>
			const Eigen::VectorXd& best_observation(const AggregatorFunction& afun = AggregatorFunction()) const
			{
				auto rewards = std::vector<double>(this->_observations[0].size());
				std::transform(this->_observations[0].begin(), this->_observations[0].end(), rewards.begin(), afun);
				auto max_e = std::max_element(rewards.begin(), rewards.end());
				return this->_observations[0][std::distance(rewards.begin(), max_e)];
			}

			/// return the best sample so far (i.e. the argmax(f(x)))
			template <typename AggregatorFunction = FirstElem>
			const Eigen::VectorXd& best_sample(const AggregatorFunction& afun = AggregatorFunction()) const
			{
				auto rewards = std::vector<double>(this->_observations[0].size());
				std::transform(this->_observations[0].begin(), this->_observations[0].end(), rewards.begin(), afun);
				auto max_e = std::max_element(rewards.begin(), rewards.end());
				return this->_samples[std::distance(rewards.begin(), max_e)];
			}


            const model_t& model() const { return _model; }
            // VALE
            const std::vector<model_t>& models_constr() const {return _models_constr;}

        protected:
            model_t _model;
            std::vector<model_t> _models_constr;
            bool _constrained;
            ParticleData _d;
            int _dim_in;
			int _dim_out;
			int _constr_dim_out;
			Eigen::VectorXd _ub;
			Eigen::VectorXd _lb;



        };

    }
}




#endif /* SRC_LIMBO_BAYES_OPT_LOCAL_ONESTEP_BOPTIMIZER_HPP_ */

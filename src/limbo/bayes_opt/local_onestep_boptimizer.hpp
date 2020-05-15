/*
 * local_onestep_boptimizer.hpp
 *
 *  Created on: May 15, 2020
 *      Author: vale
 */

// the hypothesis here is that the particles does not speak with each other


#ifndef SRC_LIMBO_BAYES_OPT_LOCAL_ONESTEP_BOPTIMIZER_HPP_
#define SRC_LIMBO_BAYES_OPT_LOCAL_ONESTEP_BOPTIMIZER_HPP_

// WARNING: local one step optimizer work only with acqui_manager as acquisition method

#include <algorithm>
#include <iostream>
#include <iterator>

#include <boost/parameter/aux_/void.hpp>

#include <Eigen/Core>

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

//std::get<0>(zoomdata)=bound,std::get<1>(zoomdata)=cur mean, std::get<2>(zoomdata)=cur covariance

namespace limbo {
    namespace defaults {
        struct local_bayes_opt_onestep_boptimizer {
            BO_PARAM(int, hp_period, -1);
        };
    }
    // nedd to manage data
    struct ParticleData{
    	Eigen::VectorXd _mean;
    	Eigen::MatrixXd _cov;
    	double          _sigma;

    	inline bool operator==(const ParticleData& lhs, const ParticleData& rhs){
    		if(lhs._mean == rhs._mean)
    			return true;
    		else
    			return false;
    	};
    	void update(const ParticleData& d){
    		// do nothing
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
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void init(const StateFunction& sfun, ParticleData& d, const AggregatorFunction& afun = AggregatorFunction()){

            	_constrained = Params::bayes_opt_bobase::constrained();
            	_d.update(d);
            	if(_models_constr.size()==0 && _constrained){
					for(uint i = 0;i<StateFunction::constr_dim_out();i++)
						_models_constr.push_back( model_t(StateFunction::dim_in(),1) );
				}
            	// initialize dimension inside the bo base class
            	this->simple_init(sfun);
            }

            // we need to call it at every new sample from (1+1-cmaes)
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
		    void update_bo(Eigen::VectorXd& sample,const StateFunction& sfun,const ParticleData& d,const AggregatorFunction& afun = AggregatorFunction()){
            	if(d == _d){ // we need to create a new local gp model
            		//update particle data
            		_d.update(d);
            		// we destroy the former models and build new ones
            		_model = model_t(StateFunction::dim_in(),StateFunction::dim_out());
            		if(_constrained){
            			_models_constr.clear();
						for(uint i = 0;i<StateFunction::constr_dim_out();i++)
							_models_constr.push_back( model_t(StateFunction::dim_in(),1) );
					}
            		// add current sample (the new mean)
            		this->eval_and_add(sfun, sample);
            		// check if i can reuse some of the older points
            		// TODO adding a function to compute the actual point inside particle data best option
            		double dist = 0;
            		init_local_model(sfun,dist);


            	}
            	else{  // we need to add the current point to the one we are working on;
            	    // update particle data
            		_d.update(d);
            		update_data(sample,sfun);
            	}
		    }

            // The main function (run the Bayesian optimization algorithm) (call it when model is sufficiently good to be used)
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            Eigen::VectorXd optimize(std::string strategy, const StateFunction& sfun,const AggregatorFunction& afun = AggregatorFunction())
            {

                acqui_optimizer_t acqui_optimizer;
                Eigen::VectorXd ub = std::get<0>(_d)*Eigen::VectorXd::Ones(StateFunction::dim_in());
                Eigen::VectorXd lb = -std::get<0>(_d)*Eigen::VectorXd::Ones(StateFunction::dim_in());

                // VALE update hyperparameters
				if (Params::bayes_opt_boptimizer::hp_period() > 0
					&& (this->_current_iteration + 1) % Params::bayes_opt_boptimizer::hp_period() == 0){
					//DEBUG
					std::cout << "updating kernel parameters" << std::endl;
					for(uint i = 0;i<this->_observations.size();i++){
						if(i == 0)
							_model.optimize_hyperparams();
						else if(_constrained)
							_models_constr[i-1].optimize_hyperparams();
					}
				}

				acquisition_function_t acqui(_model,_models_constr, strategy, this->_current_iteration);
				auto acqui_optimization = [&](const Eigen::VectorXd& x, bool g) { return acqui(rototrasl(bound_transf(x,ub,lb),std::get<1>(_d),std::get<2>(_d)), afun, g); };
				Eigen::VectorXd starting_point = tools::random_vector(StateFunction::dim_in(), Params::bayes_opt_bobase::bounded());
				Eigen::VectorXd max_sample = acqui_optimizer(acqui_optimization, starting_point, Params::bayes_opt_bobase::bounded());

				//update models
				this->eval_and_add(sfun, max_sample);
                // update models (I should not do that but I should check if the new point is feasible and is better than the current mean but
				// ignoring this in not going to be a problem because I update the data and the model and after that i create new model)
				update_data(max_sample,sfun);

				return max_sample;

            }

            // auxiliary functions------------------------------------------------------------------------------------
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void update_data(Eigen::VectorXd& sample,const StateFunction& sfun,const AggregatorFunction& afun = AggregatorFunction()){
            	// VALE update data
            	this->eval_and_add(sfun, sample);
                // value update model
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
            template <typename StateFunction>
            void init_local_model(const StateFunction& sfun,double dist){
            	std::vector<Eigen::VectorXd>  local_sample;
            	std::vector< std::vector<Eigen::VectorXd>> local_obs;
            	// initialize observations
            	if(_constrained){
					for(uint i = 0; i < StateFunction::constr_dim_out();i++){
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
            	// compute gp models
            	for(uint i = 0;i<this->_observations.size();i++){
					if(i==0)
						_model.compute(local_sample, local_obs[i]);
					else if(Params::bayes_opt_bobase::constrained()){
						_models_constr[i-1].compute(local_sample, local_obs[i]);
					}
				}

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

            // transform the bound from [0,1] to [-l,l]
            Eigen::VectorXd bound_transf(const Eigen::VectorXd& x, const Eigen::VectorXd& ub, const Eigen::VectorXd& lb){
            	Eigen::VectorXd z(x.size());
            	for(uint i = 0;i<x.size();i++){
            		z[i] = x[i]*(ub[i]-lb[i]) + lb[i];
            	}
            	return z;
            }

            //rotate the point
            Eigen::VectorXd rototrasl(const Eigen::VectorXd& x, const Eigen::VectorXd& mean, const Eigen::MatrixXd& R){
            	Eigen::VectorXd z(x.size());
            	z = mean + R*x;
            	return z;
            }



            const model_t& model() const { return _model; }
            // VALE
            const std::vector<model_t>& models_constr() const {return _models_constr;}

        protected:
            model_t _model;
            std::vector<model_t> _models_constr;
            bool _constrained;
            ParticleData _d; //= std::make_tuple(0,Eigen::VectorXd(),Eigen::MatrixXd())


        };

    }
}




#endif /* SRC_LIMBO_BAYES_OPT_LOCAL_ONESTEP_BOPTIMIZER_HPP_ */

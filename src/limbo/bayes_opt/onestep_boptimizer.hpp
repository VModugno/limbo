/*
 * onestep_boptimizer.hpp
 *
 *  Created on: May 10, 2020
 *      Author: vale
 */

// WARNING: one step optimizer work only with acqui_manager as acquisition method

#ifndef SRC_LIMBO_BAYES_OPT_ONESTEP_BOPTIMIZER_HPP_
#define SRC_LIMBO_BAYES_OPT_ONESTEP_BOPTIMIZER_HPP_

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

namespace limbo {
    namespace defaults {
        struct bayes_opt_onestep_boptimizer {
            BO_PARAM(int, hp_period, -1);
        };
    }


    namespace bayes_opt {

        using onestep_boptimizer_signature = boost::parameter::parameters<boost::parameter::optional<tag::acquiopt>,
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
        class OneStepBOptimizer : public BoBase<Params, A1, A2, A3, A4, A5, A6> {
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
            using args = typename onestep_boptimizer_signature::bind<A1, A2, A3, A4, A5, A6>::type;
            //using acquisition_function_t = typename boost::parameter::binding<args, tag::acquifun, typename defaults::acqui_t>::type;
            using acqui_optimizer_t = typename boost::parameter::binding<args, tag::acquiopt, typename defaults::acquiopt_t>::type;

            // VALE we need to initialize the empty vector of gp model just once at the beginning
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void init(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset = true){
            	_constrained = Params::bayes_opt_bobase::constrained();
            	if(_models_constr.size()==0 && _constrained){
					for(uint i = 0;i<StateFunction::constr_dim_out();i++)
						_models_constr.push_back( model_t(StateFunction::dim_in(),1) );
				}
				// get a few data samples for bootstrapping
				this->_init(sfun, afun, reset);

				// VALE initialization
				if (!this->_observations.empty()){

					//DEBUG (visualize _observations)
					/*for(uint i = 0;i<this->_observations.size();i++){
						std::cout<< "obs " << i << std::endl;
						for(uint j = 0;j<this->_observations[i].size();j++)
							std::cout<< this->_observations[i][j] <<", ";
						std::cout<< std::endl;
					}*/

					for(uint i = 0;i<this->_observations.size();i++){
						if(i==0)
							_model.compute(this->_samples, this->_observations[i]);
						else if(_constrained){
							//DEBUG
							std::cout<<"_models_constr.size()="<<_models_constr.size()<<std::endl;
							_models_constr[i-1].compute(this->_samples, this->_observations[i]);
						}
					}
				}else{
					_model = model_t(StateFunction::dim_in(), StateFunction::dim_out());
				}
            }
            /// The main function (run the Bayesian optimization algorithm)
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void optimize(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction())
            {

                acqui_optimizer_t acqui_optimizer;


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

				// VALE
                std::string strategy;
				acquisition_function_t acqui(_model,_models_constr, strategy, this->_current_iteration);

				auto acqui_optimization =
					[&](const Eigen::VectorXd& x, bool g) { return acqui(x, afun, g); };
				Eigen::VectorXd starting_point = tools::random_vector(StateFunction::dim_in(), Params::bayes_opt_bobase::bounded());
				Eigen::VectorXd max_sample = acqui_optimizer(acqui_optimization, starting_point, Params::bayes_opt_bobase::bounded());

				// VALE update models
				this->eval_and_add(sfun, max_sample);

				for(uint i = 0;i<this->_observations.size();i++){
					if(i == 0)
						_model.add_sample(this->_samples.back(), this->_observations[i].back());
					else if(_constrained)
						_models_constr[i-1].add_sample(this->_samples.back(), this->_observations[i].back());
				}

				//update stats
				this->_update_stats(*this, afun);

				// DEBUG
				std::cout << "model params = " << _model.kernel_function().h_params().transpose() << std::endl;
				if(_constrained){
					for(uint jj =0 ; jj< _models_constr.size(); jj++ )
						std::cout << "constr["<<jj<<"] params = "<< _models_constr[jj].kernel_function().h_params().transpose()<<std::endl;
				}


				this->_current_iteration++;
				this->_total_iterations++;

            }
            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void update_data(Eigen::VectorXd& sample,const StateFunction& sfun,const AggregatorFunction& afun = AggregatorFunction()){
            	// VALE update models
            	this->eval_and_add(sfun, sample);

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
            /*~BOptimizer(){
            	std::cout << "-2 "<< std::endl;
            	//_models_constr.clear();
            	std::cout << "-1 "<< std::endl;
            };*/

        protected:
            model_t _model;
            std::vector<model_t> _models_constr;
            bool _constrained;

        };

    }
}






#endif /* SRC_LIMBO_BAYES_OPT_ONESTEP_BOPTIMIZER_HPP_ */

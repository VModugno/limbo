/*
 * acqui_manager.hpp
 *
 *  Created on: May 11, 2020
 *      Author: vale
 */

#ifndef SRC_LIMBO_ACQUI_ACQUI_MANAGER_HPP_
#define SRC_LIMBO_ACQUI_ACQUI_MANAGER_HPP_

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <Eigen/Core>
#include <vector>
#include <math.h>
#include <limbo/acqui.hpp>



namespace limbo {

    namespace acqui {
        /** @ingroup acqui




        Parameters:
          - `double delta`
        \endrst
        */
        template <typename Params,typename Model>
        class AcquiManager {
        public:


        	AcquiManager(const Model& model, const std::vector<Model>&  model_constr, std::string strategy, int iteration = 0)
           {
        		_dim_in   = model.dim_in();
        		_dim_out  = model.dim_out();
        		_strategy = strategy;
        		// adding here elements
        		if(_strategy.compare("eci") == 0){
        			_eci = std::make_shared< acqui::ECI<Params, Model> >(model,model_constr,iteration);
        		}
        		else if(_strategy.compare("ucb") == 0){
        			_ucb = std::make_shared< acqui::UCB<Params, Model> >(model,model_constr,iteration);
        		}else{
        			std::cerr <<"specify a correct strategy!"<<std::endl;
        			assert(false);
        		}

           }
           /*~AcquiManager(){
        	   if(_strategy.compare("eci") == 0){
					delete _eci;
				}
				else if(_strategy.compare("ucb") == 0){
					delete _ucb;
				}

           }*/

            size_t dim_in() const { return _dim_in; }

            size_t dim_out() const { return _dim_out; }


            template <typename AggregatorFunction>
            opt::eval_t operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun, bool gradient)
            {
            	// adding here elements
            	if(_strategy.compare("eci") == 0){
					return (*_eci)(v,afun,gradient);
				}
				else if(_strategy.compare("ucb") == 0){
					return (*_ucb)(v,afun,gradient);
				}
            }



        protected:
            int _dim_in;
            int _dim_out;
            std::string _strategy;
            // list of acquisition strategy classes
            std::shared_ptr<acqui::UCB<Params, Model>> _ucb;
            std::shared_ptr<acqui::ECI<Params, Model>> _eci;



        };
    }
}


#endif /* SRC_LIMBO_ACQUI_ACQUI_MANAGER_HPP_ */

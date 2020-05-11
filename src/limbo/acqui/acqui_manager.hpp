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


        	AcquiManager(const Model& model,const std::vector<Model>&  model_constr,std::string strategy,int iteration = 0)
           {
        		_dim_in  = model.dim_in();
        		_dim_out = model.dim_out();
        		_strategy = strategy;

        		if(_strategy.compare("eci") == 0){
        			_eci = acqui::ECI<Params, Model>(model,model_constr,iteration);
        		}
        		else if(_strategy.compare("ucb") == 0){
        			_ucb = acqui::UCB<Params, Model>(model,model_constr,iteration);
        		}else{
        			std::cout<<"specify a correct strategy!"<<std::endl;
        		}

           }

            size_t dim_in() const { return _dim_in; }

            size_t dim_out() const { return _dim_out; }


            template <typename AggregatorFunction>
            opt::eval_t operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun, bool gradient)
            {
            	if(_strategy.compare("eci") == 0){
					return _eci(v,afun,gradient);
				}
				else if(_strategy.compare("ucb") == 0){
					return _ucb(v,afun,gradient);
				}else{
					std::cout<<"specify a correct strategy!"<<std::endl;
				}

            }

            // specify destructor

        protected:
            int _dim_in;
            int _dim_out;
            std::string _strategy;
            // list of acquisition strategy classes
            acqui::UCB<Params, Model> _ucb;
            acqui::ECI<Params, Model> _eci;



        };
    }
}


#endif /* SRC_LIMBO_ACQUI_ACQUI_MANAGER_HPP_ */

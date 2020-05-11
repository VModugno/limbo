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
#include <limbo/acqui/eci.hpp>
#include <limbo/acqui/ei.hpp>
#include <limbo/acqui/gp_ucb.hpp>
#include <limbo/bayes_opt/boptimizer.hpp>
#include <limbo/bayes_opt/onestep_boptimizer.hpp>
#include <limbo/kernel/matern_five_halves.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/stat.hpp>
#include <limbo/tools/macros.hpp>

using namespace limbo;

// where are the paramaters bounds?     (bounded between 0 and 1)
// where the gp model is constructed?   (it is built by using the default constructor)


BO_PARAMS(std::cout,
          struct Params {

#ifdef USE_NLOPT
              struct opt_nloptnograd : public defaults::opt_nloptnograd {

              };
              struct acqui_eci : public defaults::acqui_eci {
              	  BO_PARAM(double, xi, 0.01);
              };
              struct acqui_ei {
              	  BO_PARAM(double, jitter, 0.01);
			  };
#elif defined(USE_LIBCMAES)
              struct opt_cmaes : public defaults::opt_cmaes {
              };
#else
              struct opt_gridsearch : public defaults::opt_gridsearch {
              };
#endif
              struct acqui_ucb {
			  	  BO_PARAM(double, alpha, 0.1);
			  };
              struct kernel : public defaults::kernel {
                  BO_PARAM(double, noise, 0.00001);
              };

              struct kernel_maternfivehalves {
                  BO_PARAM(double, sigma_sq, 10);
                  BO_PARAM(double, l, 0.2);
              };
              struct kernel_exp : public defaults::kernel_exp {
              };
              struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
                  BO_PARAM(bool, stats_enabled, true);
                  BO_PARAM(bool, constrained, true);
              };

              struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
            	  BO_PARAM(int, hp_period, 1);
              };

              struct init_randomsampling {
                  BO_PARAM(int, samples, 20);
              };

              struct stop_maxiterations {
                  BO_PARAM(int, iterations, 30);
              };
              struct stat_gp {
                  BO_PARAM(int, bins, 20);
              };

              struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
              };

              struct opt_rprop : public defaults::opt_rprop {
              };
          };)



struct fit_eval {
    BO_PARAM(size_t, dim_in, 2);
    BO_PARAM(size_t, dim_out, 1);
    BO_PARAM(size_t, constr_dim_out, 2); // each constraints is considered a mapping R^n -> R

    Eigen::VectorXd operator()(const Eigen::VectorXd& z) const
    {
    	Eigen::VectorXd res(this->dim_out()+this->constr_dim_out());
    	Eigen::VectorXd x(this->dim_in());
    	// bound manager to denormalize data      z = (x-min)/ (max-min)     x = z*(max-min) + min
    	// 13<x1<100; 0<x2<100
    	x(0) = z(0)*(100-13) + 13; x(1) = z(1)*(100);
    	//std::cout<< "current_sample = "<< x <<std::endl;
        // fitness
    	double res_fit;
        res_fit =  -( pow((x(0) - 10),3) + pow((x(1) - 20),3) );

        // constraints have all the same expression = f(x) < 0;
        double res_constr1 = -pow((x(0)-5),2) -pow((x(1)-5),2) + 100;
        double res_constr2 = +pow((x(0)-6),2) +pow((x(1)-5),2) - 82.81;
        // the BO expect in the first dim_out positions of res the reward functions and after that
        // all the constraints
        res(0) = res_fit; res(1) = res_constr1; res(2) = res_constr2;
        return res;

    }
};

/*struct fit_eval {
    BO_PARAM(size_t, dim_in, 7);
    BO_PARAM(size_t, dim_out, 1);
    BO_PARAM(size_t, constr_dim_out, 4); // each constraints is considered a mapping R^n -> R

    Eigen::VectorXd operator()(const Eigen::VectorXd& z) const
    {
    	Eigen::VectorXd res(this->dim_out()+this->constr_dim_out());
    	Eigen::VectorXd x(this->dim_in());
    	// bound manager to denormalize data      z = (x-min)/ (max-min)     x = z*(max-min) + min
    	// 13<x1<100; 0<x2<100
    	x(0) = z(0)*(10+10) -10; x(1) = z(1)*(10+10) -10; x(2) = z(2)*(10+10) -10;
    	x(3) = z(3)*(10+10) -10; x(4) = z(4)*(10+10) -10; x(5) = z(5)*(10+10) -10;
    	x(6) = z(6)*(10+10) -10;
    	//std::cout<< "current_sample = "<< x <<std::endl;
        // fitness
    	double res_fit;
        res_fit =  -( pow((x(0) - 10),2) + pow((x(1) - 12),2) + pow((x(2)),4) + 3*pow((x(3) - 11),2) + 10*pow(x(4),6) + 7*pow(x(5),2) + pow(x(6),4)  - 4*x(5)*x(6) - 10*x(5) -8*x(6));

        // constraints have all the same expression = f(x) < 0;
        double res_constr1 = -127 + 2*pow(x(0),2) + 3*pow(x(1),4) + x(2) + 4*pow(x(3),2) + 5*x(4);
        double res_constr2 = -196 + 23*x(0) + pow(x(1),2) + 6*pow(x(5),2) - 8*x(6);
        double res_constr3 = -282 + 7*x(0) + 3*x(1) + 10*pow(x(2),2) + x(3) -x(4);
		double res_constr4 = 4*pow(x(0),2) + pow(x(1),2) -3*x(0)*x(1) + 2*pow(x(2),2) + 5*x(5) -11*x(6);
        // the BO expect in the first dim_out positions of res the reward functions and after that
        // all the constraints
        res(0) = res_fit; res(1) = res_constr1; res(2) = res_constr2; res(3) = res_constr3; res(4) = res_constr4;
        return res;

    }
};*/


/*struct fit_eval {
    BO_PARAM(size_t, dim_in, 2);
    BO_PARAM(size_t, dim_out, 1);
    BO_PARAM(size_t, constr_dim_out, 0);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        double res = 0;
        for (int i = 0; i < x.size(); i++)
            res += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
        return tools::make_vector(res);
    }
};*/



int main()
{   //using Kernel_t = kernel::SquaredExpARD<Params>;
    using Kernel_t = kernel::MaternFiveHalves<Params>;
	//using Kernel_t = kernel::SquaredExpARD<Params>;
    using Mean_t   = mean::Data<Params>;
    using gp_opt_t = model::gp::KernelLFOpt<Params>;
    using GP_t     = model::GP<Params, Kernel_t, Mean_t, gp_opt_t>;
    //using Acqui_t = acqui::EI<Params, GP_t>;
    //using Acqui_t = acqui::ECI<Params, GP_t>;
    using Acqui_t = acqui::UCB<Params, GP_t>;
    using Acqui_t_one_step = acqui::AcquiManager<Params, GP_t>;
    using stat_t = boost::fusion::vector<stat::ConsoleSummary<Params>,
        stat::Samples<Params>,
        stat::Observations<Params>>;
        //stat::GP<Params>>;
    std::string strategy = "eci";

    bayes_opt::OneStepBOptimizer <Params, modelfun<GP_t>, statsfun<stat_t>,acquifun<Acqui_t_one_step>> opt_one_step;
    //bayes_opt::BOptimizer<Params, modelfun<GP_t>, statsfun<stat_t>, acquifun<Acqui_t>> opt;
    //opt.optimize(fit_eval());
    opt_one_step.init(fit_eval());
    opt_one_step.optimize(strategy,fit_eval());
    Eigen::VectorXd x_best = opt_one_step.best_sample();
    /*x_best(0) = x_best(0)*(100-13) + 13; x_best(1) = x_best(1)*(100);
    std::cout << opt.best_observation() << " res  "
              << x_best.transpose() << std::endl;*/
    std::cout << opt_one_step.best_observation() << " res  "
              << opt_one_step.best_sample().transpose() << std::endl;

    return 0;
}

//// Created by Huy Vo on 6/29/2019.//static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";#include <gtest/gtest.h>#include "SensFspSolverMultiSinks.h"#include "pacmensl_test_env.h"namespace toggle_cme {/* Stoichiometric matrix of the toggle switch model */arma::Mat<PetscInt> SM{{1, 1, -1, 0, 0, 0},                       {0, 0, 0, 1, 1, -1}};const int nReaction = 6;/* Parameters for the propensity functions */const double ayx{2.6e-3}, axy{6.1e-3}, nyx{3.0e0}, nxy{2.1e0}, kx0{2.2e-3}, kx{1.7e-2}, dx{3.8e-4}, ky0{6.8e-5}, ky{    1.6e-2}, dy{3.8e-4};// Function to constraint the shape of the Fspvoid lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states, int *vals,                void *args){  for (int i{0}; i < num_states; ++i)  {    vals[i * num_constrs]     = states[num_species * i];    vals[i * num_constrs + 1] = states[num_species * i + 1];    vals[i * num_constrs + 2] = states[num_species * i] * states[num_species * i + 1];  }}arma::Row<int>    rhs_constr{200, 200, 2000};arma::Row<double> expansion_factors{0.2, 0.2, 0.2};// propensity function for toggleint propensity(const int reaction, const int num_species, const int num_states, const PetscInt *X, double *outputs,               void *args){  int (*X_view)[2] = ( int (*)[2] ) X;  switch (reaction)  {    case 0:for (int i{0}; i < num_states; ++i) { outputs[i] = 1.0; }      break;    case 1:      for (int i{0}; i < num_states; ++i)      {        outputs[i] = 1.0 / (1.0 + ayx * pow(PetscReal(X_view[i][1]), nyx));      }      break;    case 2:for (int i{0}; i < num_states; ++i) { outputs[i] = PetscReal(X_view[i][0]); }      break;    case 3:for (int i{0}; i < num_states; ++i) { outputs[i] = 1.0; }      break;    case 4:      for (int i{0}; i < num_states; ++i)      {        outputs[i] = 1.0 / (1.0 + axy * pow(PetscReal(X_view[i][0]), nxy));      }      break;    case 5:for (int i{0}; i < num_states; ++i) { outputs[i] = PetscReal(X_view[i][1]); }      break;    default:return -1;  }  return 0;}int t_fun(PetscReal t, int n_coefs, double *outputs, void *args){  outputs[0] = kx0;  outputs[1] = kx;  outputs[2] = dx;  outputs[3] = ky0;  outputs[4] = ky;  outputs[5] = dy;  return 0;}int get_sens_t_fun(int i, PetscReal t, int n_coefs, double *outputs, void *args){  outputs[i] = 1.0;  return 0;}}using namespace pacmensl;class SensFspTest : public ::testing::Test{ protected:  SensFspTest() {}  void SetUp() override  {    int n_par = 6;    t_final = 100.0;    fsp_tol = 1.0e-10;    X0      = X0.t();    dp0     = std::vector<arma::Col<PetscReal>>(n_par, arma::Col<PetscReal>({0.0}));    std::vector<TcoefFun> d_t_fun(n_par);    for (int              i{0}; i < n_par; ++i)    {      d_t_fun[i] = std::bind(toggle_cme::get_sens_t_fun,                             i,                             std::placeholders::_1,                             std::placeholders::_2,                             std::placeholders::_3,                             std::placeholders::_4);    }    toggle_model = SensModel(toggle_cme::SM,                             std::vector<int>({0,1,2,3,4,5}),                             toggle_cme::t_fun,                             toggle_cme::propensity,                             d_t_fun,                             std::vector<PropFun>(n_par, toggle_cme::propensity),                             std::vector<int>({0, 1, 2, 3, 4, 5}),                             std::vector<int>({0, 1, 2, 3, 4, 5, 6}));  }  void TearDown() override  {  }  PetscReal                         t_final, fsp_tol;  arma::Mat<PetscInt>               X0{0, 0};  arma::Col<PetscReal>              p0   = {1.0};  std::vector<arma::Col<PetscReal>> dp0;  SensModel            toggle_model;  arma::Row<int>       fsp_size          = {1, 1};  arma::Row<PetscReal> expansion_factors = {0.25, 0.25};};TEST_F(SensFspTest, test_wrong_call_sequence_detection){  int                     ierr;  SensFspSolverMultiSinks fsp(PETSC_COMM_WORLD);  ierr = fsp.SetUp();  ASSERT_EQ(ierr, -1);}TEST_F(SensFspTest, test_handling_t_fun_error){  int ierr;  SensFspSolverMultiSinks fsp(PETSC_COMM_WORLD);  std::vector<PetscReal>  tspan     =                              arma::conv_to<std::vector<PetscReal>>::from(arma::linspace<arma::Row<PetscReal>>(                                  0.0,                                  t_final,                                  3));  SensModel               bad_model = toggle_model;  bad_model.prop_t_ = [&](double t, int n, double *vals, void *args) {    return -1;  };  ierr = fsp.SetModel(bad_model);  ASSERT_FALSE(ierr);  ierr = fsp.SetInitialBounds(fsp_size);  ASSERT_FALSE(ierr);  ierr = fsp.SetExpansionFactors(expansion_factors);  ASSERT_FALSE(ierr);  ierr = fsp.SetVerbosity(0);  ASSERT_FALSE(ierr);  ierr = fsp.SetInitialDistribution(X0, p0, dp0);  ASSERT_FALSE(ierr);  ierr = fsp.SetUp();  ASSERT_FALSE(ierr);  ASSERT_THROW(fsp.Solve(t_final, fsp_tol), std::runtime_error);  fsp.ClearState();}TEST_F(SensFspTest, toggle_sens_solve_with_cvode){  PetscInt                              ierr;  PetscReal                             stmp;  SensDiscreteDistribution              p_final_bdf;  std::vector<SensDiscreteDistribution> p_snapshots_bdf;  std::vector<PetscReal>                tspan;  Vec                                   q;  tspan = arma::conv_to<std::vector<PetscReal>>::from(arma::linspace<arma::Row<PetscReal>>(0.0, t_final, 3));  SensFspSolverMultiSinks fsp(PETSC_COMM_WORLD);  ierr = fsp.SetModel(toggle_model);  ASSERT_FALSE(ierr);  ierr = fsp.SetInitialBounds(fsp_size);  ASSERT_FALSE(ierr);  ierr = fsp.SetExpansionFactors(expansion_factors);  ASSERT_FALSE(ierr);  ierr = fsp.SetInitialDistribution(X0, p0, dp0);  ASSERT_FALSE(ierr);  ierr = fsp.SetUp();  ASSERT_FALSE(ierr);  p_final_bdf = fsp.Solve(t_final, fsp_tol);  fsp.ClearState();  ierr = VecSum(p_final_bdf.p_, &stmp);  ASSERT_FALSE(ierr);  ASSERT_GE(stmp, 1.0-fsp_tol);  for (int i{0}; i < 6; ++i)  {    ierr = VecSum(p_final_bdf.dp_[i], &stmp);    ASSERT_FALSE(ierr);    ASSERT_LE(abs(stmp), 1.0e-6);  }//  arma::Mat<PetscReal> FIM;//  ComputeFIM(p_final_bdf, FIM);//  std::cout << FIM;}class SensFspPoissonTest : public ::testing::Test{ protected:  SensFspPoissonTest() {}  void SetUp() override  {    auto propensity =             [&](int reaction, int num_species, int num_states, const int *state, PetscReal *output, void *args) {               for (int i{0}; i < num_states; ++i)               {                 output[i] = 1.0;               }               return 0;             };    auto t_fun      = [&](double t, int num_coefs, double *outputs, void *args) {      outputs[0] = lambda;      return 0;    };    auto d_t_fun    = [&](double t, int num_coefs, double *outputs, void *args) {      outputs[0] = 1.0;      return 0;    };    poisson_model = SensModel(stoich_matrix,                              std::vector<int>({0}),                              t_fun,                              propensity,                              std::vector<TcoefFun>({d_t_fun}),                              std::vector<PropFun>({propensity}));  }  void TearDown() override  {  }  ~SensFspPoissonTest() {}  SensModel            poisson_model;  PetscReal            lambda            = 2.0;  arma::Mat<int>       stoich_matrix     = {1};  arma::Mat<int>       x0                = {0};  arma::Col<PetscReal> p0                = {1.0};  arma::Col<PetscReal> s0                = {0.0};  arma::Row<int>       fsp_size          = {5};  arma::Row<PetscReal> expansion_factors = {0.1};  PetscReal            t_final{1.0}, fsp_tol{1.0e-7};};TEST_F(SensFspPoissonTest, test_poisson_analytic){  PetscInt                 ierr;  PetscReal                stmp;  SensDiscreteDistribution p_final;  SensFspSolverMultiSinks fsp(PETSC_COMM_WORLD);  ierr = fsp.SetModel(poisson_model);  ASSERT_FALSE(ierr);  ierr = fsp.SetInitialBounds(fsp_size);  ASSERT_FALSE(ierr);  ierr = fsp.SetExpansionFactors(expansion_factors);  ASSERT_FALSE(ierr);  ierr = fsp.SetInitialDistribution(x0, p0, std::vector<arma::Col<PetscReal>>({s0}));  ASSERT_FALSE(ierr);  ierr = fsp.SetUp();  ASSERT_FALSE(ierr);  p_final = fsp.Solve(t_final, fsp_tol);  fsp.ClearState();  // Check that the solution is close to Poisson  stmp        = 0.0;  PetscReal *p_dat;  int num_states;  p_final.GetProbView(num_states, p_dat);  PetscReal pdf;  int       n;  for (int  i = 0; i < num_states; ++i)  {    n   = p_final.states_(0, i);    pdf = exp(-lambda * t_final) * pow(lambda * t_final, double(n)) / tgamma(n + 1);    stmp += abs(p_dat[i] - pdf);  }  p_final.RestoreProbView(p_dat);  MPI_Allreduce(&stmp, MPI_IN_PLACE, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);  ASSERT_LE(stmp, fsp_tol);  PetscReal *s_dat;  p_final.GetSensView(0, num_states, s_dat);  for (int  i = 0; i < num_states; ++i)  {    n   = p_final.states_(0, i);    pdf = -t_final*exp(-lambda*t_final)*pow(lambda*t_final, double(n))/tgamma(n+1)         + exp(-lambda*t_final)*t_final*pow(lambda*t_final, double(n-1))/tgamma(n);    stmp += abs(s_dat[i] - pdf);  }  p_final.RestoreSensView(0, s_dat);  MPI_Allreduce(&stmp, MPI_IN_PLACE, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD);  PetscPrintf(PETSC_COMM_WORLD, "Sensitivity error = %.2e \n", stmp);  ASSERT_LE(stmp, 1.0e-6);}
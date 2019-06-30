//// Created by Huy Vo on 12/6/18.//static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";#include <gtest/gtest.h>#include "pacmensl_all.h"#include "SensFspSolverMultiSinks.h"#include "pacmensl_test_env.h"namespace toggle_cme {/* Stoichiometric matrix of the toggle switch model */arma::Mat<PetscInt> SM{{1, 1, -1, 0, 0, 0},                       {0, 0, 0, 1, 1, -1}};const int nReaction = 6;/* Parameters for the propensity functions */const double ayx{2.6e-3}, axy{6.1e-3}, nyx{3.0e0}, nxy{2.1e0}, kx0{2.2e-3}, kx{1.7e-2}, dx{3.8e-4}, ky0{6.8e-5}, ky{    1.6e-2}, dy{3.8e-4};// Function to constraint the shape of the Fspvoid lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states, int *vals,                void *args) {  for (int i{0}; i < num_states; ++i) {    vals[i * num_constrs]     = states[num_species * i];    vals[i * num_constrs + 1] = states[num_species * i + 1];    vals[i * num_constrs + 2] = states[num_species * i] * states[num_species * i + 1];  }}arma::Row<int>    rhs_constr{200, 200, 2000};arma::Row<double> expansion_factors{0.2, 0.2, 0.2};// propensity function for toggleint propensity(const int reaction, const int num_species, const int num_states, const PetscInt *X, double *outputs,               void *args) {  int (*X_view)[2] = ( int (*)[2] ) X;  switch (reaction) {    case 0:for (int i{0}; i < num_states; ++i) { outputs[i] = 1.0; }      break;    case 1:      for (int i{0}; i < num_states; ++i) {        outputs[i] = 1.0 / (1.0 + ayx * pow(PetscReal(X_view[i][1]), nyx));      }      break;    case 2:for (int i{0}; i < num_states; ++i) { outputs[i] = PetscReal(X_view[i][0]); }      break;    case 3:for (int i{0}; i < num_states; ++i) { outputs[i] = 1.0; }      break;    case 4:      for (int i{0}; i < num_states; ++i) {        outputs[i] = 1.0 / (1.0 + axy * pow(PetscReal(X_view[i][0]), nxy));      }      break;    case 5:for (int i{0}; i < num_states; ++i) { outputs[i] = PetscReal(X_view[i][1]); }      break;    default:return -1;  }  return 0;}int t_fun(PetscReal t, int n_coefs, double *outputs, void *args) {  outputs[0] = kx0;  outputs[1] = kx;  outputs[2] = dx;  outputs[3] = ky0;  outputs[4] = ky;  outputs[5] = dy;  return 0;}int get_sens_t_fun(int i, PetscReal t, int n_coefs, double *outputs, void *args) {  outputs[0] = 0.0;  outputs[1] = 0.0;  outputs[2] = 0.0;  outputs[3] = 0.0;  outputs[4] = 0.0;  outputs[5] = 0.0;  outputs[i] = 1.0;  return 0;}}using namespace pacmensl;class SensFspTest : public ::testing::Test { protected:  SensFspTest() {}  void SetUp() override {    int n_par = 6;    t_final = 100.0;    fsp_tol = 1.0e-6;    X0      = X0.t();    dp0     = std::vector<arma::Col<PetscReal>>(n_par, arma::Col<PetscReal>({0.0}));    std::vector<TcoefFun> d_t_fun(n_par);    for (int              i{0}; i < n_par; ++i) {      d_t_fun[i] = std::bind(toggle_cme::get_sens_t_fun, i, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);    }    toggle_model = SensModel(toggle_cme::SM,                             toggle_cme::t_fun,                             nullptr,                             toggle_cme::propensity,                             nullptr,                             d_t_fun,                             std::vector<void *>(n_par, nullptr),                             std::vector<PropFun>(n_par, toggle_cme::propensity),                             std::vector<void *>(n_par, nullptr),                             arma::Mat<char>(0, 0));  }  void TearDown() override {  }  PetscReal                         t_final, fsp_tol;  arma::Mat<PetscInt>               X0{0, 0};  arma::Col<PetscReal>              p0   = {1.0};  std::vector<arma::Col<PetscReal>> dp0;  SensModel            toggle_model;  arma::Row<int>       fsp_size          = {1, 1};  arma::Row<PetscReal> expansion_factors = {0.25, 0.25};};TEST_F(SensFspTest, test_wrong_call_sequence_detection) {  int                     ierr;  SensFspSolverMultiSinks fsp(PETSC_COMM_WORLD);  ierr = fsp.SetUp();  ASSERT_EQ(ierr, -1);}//TEST_F(SensFspTest, test_handling_t_fun_error) {//  int                               ierr;////  SensFspSolverMultiSinks               fsp(PETSC_COMM_WORLD);//  std::vector<PetscReal>            tspan     =//                                        arma::conv_to<std::vector<PetscReal>>::from(arma::linspace<arma::Row<PetscReal>>(//                                            0.0,//                                            t_final,//                                            3));//  SensModel                             bad_model = toggle_model;//  bad_model.propensity_tfac_ = [&](double t, int n, double *vals, void *args) {//    return -1;//  };////  ierr = fsp.SetModel(bad_model);//  ASSERT_FALSE(ierr);//  ierr = fsp.SetInitialBounds(fsp_size);//  ASSERT_FALSE(ierr);//  ierr = fsp.SetExpansionFactors(expansion_factors);//  ASSERT_FALSE(ierr);//  ierr = fsp.SetVerbosity(0);//  ASSERT_FALSE(ierr);//  ierr = fsp.SetInitialDistribution(X0, p0, dp0);//  ASSERT_FALSE(ierr);////  ierr = fsp.SetUp();//  ASSERT_FALSE(ierr);//  ASSERT_THROW(fsp.Solve(t_final, fsp_tol), std::runtime_error);//  fsp.ClearState();//}TEST_F(SensFspTest, toggle_sens_solve_with_cvode) {  PetscInt                              ierr;  PetscReal                             stmp;  SensDiscreteDistribution              p_final_bdf;  std::vector<SensDiscreteDistribution> p_snapshots_bdf;  std::vector<PetscReal>                tspan;  Vec                                   q;  tspan = arma::conv_to<std::vector<PetscReal>>::from(arma::linspace<arma::Row<PetscReal>>(0.0, t_final, 3));  SensFspSolverMultiSinks fsp(PETSC_COMM_WORLD);  ierr = fsp.SetModel(toggle_model);  ASSERT_FALSE(ierr);  ierr = fsp.SetInitialBounds(fsp_size);  ASSERT_FALSE(ierr);  ierr = fsp.SetExpansionFactors(expansion_factors);  ASSERT_FALSE(ierr);  ierr = fsp.SetInitialDistribution(X0, p0, dp0);  ASSERT_FALSE(ierr);  ierr = fsp.SetUp();  ASSERT_FALSE(ierr);  fsp.SetVerbosity(2);  p_final_bdf     = fsp.Solve(t_final, fsp_tol);  fsp.ClearState();  ierr = VecSum(p_final_bdf.p_, &stmp);  ASSERT_FALSE(ierr);  for (int i{0}; i < 6; ++i){   ierr = VecSum(p_final_bdf.dp_[i], &stmp);   std::cout << stmp << "\n";  }}
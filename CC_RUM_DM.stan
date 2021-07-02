
data {
  int N; // number of rows
  int I; // number of individuals
  int Kx; // number of columns in X
  int Kz; // number of columns in Z
  int T; // number of individual-choice set combination
  vector<lower = 0, upper = 1>[N] y; // choice indicator
  
  real phi_prior_mean; // sensitivity analysis
  real phi_prior_sd; //
  
  matrix[N,Kx] X; // attribute matrix
  matrix[I, Kz] Z; // sociodemographic matrix
  int person[N]; // index for individual
  
  int start[T]; // the starting observation for each choice set
  int end[T]; // the ending observation for each choice set
  vector[N] action; // action indicator

  matrix[N,2] trX; // attribute matrix
  matrix[N,2] gfX; // attribute matrix
  matrix[N,2] bdX; // attribute matrix
  matrix[N,2] plX; // attribute matrix
  matrix[N,2] wqX; // attribute matrix
  matrix[N,2] priceX; // attribute matrix  
  
  vector[N] max_price; // max price in choice set
}

parameters {
  vector<lower = 0>[5] CCbeta;
  real<lower = 0> CCbeta_price;
  real CCdelta;
  real<lower = 0> phi_price;
  real<lower = 0> phi_tr;
  real<lower = 0> phi_gf;
  real<lower = 0> phi_bd;
  real<lower = 0> phi_pl;
  real<lower = 0> phi_wq;

  vector<lower = 0>[5] RUMbeta;
  real<upper = 0> RUMbeta_price;
  real RUMdelta;
    vector[Kz] gamma;
  //real mu_alpha;
  //vector[I] alpha;
  //real<lower = 0> sigma_alpha;
}

transformed parameters {
  vector[N] CCMlinpred; 
  vector[N] CCMlog_prob;
  vector[N] RUMlog_prob;
  vector[N] RUMlinpred; 
    //vector[T]  RUM_ind_probs ;
  vector[I] rho;

  vector[T] log_lik;
    
  for(idx in 1:N){
  CCMlinpred[idx]  = action[idx]*CCdelta +
                        CCbeta[1]*(X[idx,1]^phi_tr)+ 
                        CCbeta[2]*(X[idx,2]^phi_gf)+ 
                        CCbeta[3]*(X[idx,3]^phi_bd)+ 
                        CCbeta[4]*(X[idx,4]^phi_pl)+ 
                        CCbeta[5]*(X[idx,5]^phi_wq)+
                        CCbeta_price*((max_price[idx]-X[idx,6] )^phi_price); 
        // print(phi_price);                    
  RUMlinpred[idx]  = action[idx]*RUMdelta +
                        (X[idx,1])*RUMbeta[1] + 
                        (X[idx,2])*RUMbeta[2] + 
                        (X[idx,3])*RUMbeta[3] + 
                        (X[idx,4])*RUMbeta[4] + 
                        (X[idx,5])*RUMbeta[5] +
                        (X[idx,6])*RUMbeta_price; 
  }                      
  
  for (i in 1:I){
    rho[i] = inv_logit( Z[i,]*gamma );//++ alpha[i] 
  }
 
 
   for(i in 1:T) {
    CCMlog_prob[start[i]:end[i]] = log_softmax(CCMlinpred[start[i]:end[i]]);
    RUMlog_prob[start[i]:end[i]] =  log_softmax(RUMlinpred[start[i]:end[i]]);
    log_lik[i]= log_mix(rho[person[start[i]]],
              dot_product(CCMlog_prob[start[i]:end[i]],y[start[i]:end[i]]),
              dot_product(RUMlog_prob[start[i]:end[i]],y[start[i]:end[i]]));
   }
  
}

model {  
    CCdelta ~  normal(0,1);
    RUMdelta ~  normal(0,1);
    CCbeta ~  normal(0, 1);
    RUMbeta ~  normal(0, 1);
    CCbeta_price ~ normal(0, 1);
    RUMbeta_price ~ normal(0, 1);
   // mu_alpha ~ normal(0,1);
   // alpha ~ normal(0,sigma_alpha);
    //sigma_alpha ~ normal(0, .5);
        gamma ~ normal(0, 0.5);
    
    phi_tr ~ normal(phi_prior_mean, phi_prior_sd);
    phi_gf ~ normal(phi_prior_mean, phi_prior_sd);
    phi_bd ~ normal(phi_prior_mean, phi_prior_sd);
    phi_pl ~ normal(phi_prior_mean, phi_prior_sd);
    phi_wq ~ normal(phi_prior_mean, phi_prior_sd);
    phi_price ~ normal(phi_prior_mean, phi_prior_sd);
    
  // log probabilities of each choice in the dataset
  for(i in 1:T) {
    target += log_lik[i];
  }
  

}

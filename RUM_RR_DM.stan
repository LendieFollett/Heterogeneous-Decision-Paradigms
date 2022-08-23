
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
  vector<lower = 0>[5]  RRbeta;
  real<lower = 0>  RRmu;
  vector<lower = 0>[5]  RUMbeta;
  real<upper = 0> RRbeta_price;
  real<upper = 0> RUMbeta_price;
  real RRdelta;
  real RUMdelta;
  vector[Kz] gamma;
  //vector[I] alpha;
  //real<lower = 0> sigma_alpha;
  //real mu_alpha;
}

transformed parameters {
    vector[N] RUMlog_prob;
    vector[N] RRlog_prob;
    //vector[T]  RUM_ind_probs ;
  vector[I] rho;
  vector[N] RUMlinpred; 
  vector[N] RRlinpred; 
  vector[T] log_lik;
  for(idx in 1:N){
  RUMlinpred[idx]  = action[idx]*RUMdelta +
                        (X[idx,1])*RUMbeta[1] + 
                        (X[idx,2])*RUMbeta[2] + 
                        (X[idx,3])*RUMbeta[3] + 
                        (X[idx,4])*RUMbeta[4] + 
                        (X[idx,5])*RUMbeta[5] +
                        (X[idx,6])*RUMbeta_price; 
        // print(phi_price);                    
  RRlinpred[idx]  =  action[idx]*RRdelta +
            RRmu*log(1 + exp(trX[idx,1]*RRbeta[1]/RRmu)) + 
            RRmu*log(1 + exp(trX[idx,2]*RRbeta[1]/RRmu))+                   
            RRmu*log(1 + exp(gfX[idx,1]*RRbeta[2]/RRmu)) + 
            RRmu*log(1 + exp(gfX[idx,2]*RRbeta[2]/RRmu))+ 
            RRmu*log(1 + exp(bdX[idx,1]*RRbeta[3]/RRmu)) + 
            RRmu*log(1 + exp(bdX[idx,2]*RRbeta[3]/RRmu))+ 
            RRmu*log(1 + exp(plX[idx,1]*RRbeta[4]/RRmu)) + 
            RRmu*log(1 + exp(plX[idx,2]*RRbeta[4]/RRmu))+ 
            RRmu*log(1 + exp(wqX[idx,1]*RRbeta[5]/RRmu)) + 
            RRmu*log(1 + exp(wqX[idx,2]*RRbeta[5]/RRmu))+
            RRmu*log(1 + exp(priceX[idx,1]*RRbeta_price/RRmu)) + 
            RRmu*log(1 + exp(priceX[idx,2]*RRbeta_price/RRmu)) ;
  }                      
  
  for (i in 1:I){
    rho[i] = inv_logit( Z[i,]*gamma );//++ alpha[i] 
  }
 
 
   for(i in 1:T) {
    RUMlog_prob[start[i]:end[i]] = log_softmax(RUMlinpred[start[i]:end[i]]);
    RRlog_prob[start[i]:end[i]] = log_softmax(-RRlinpred[start[i]:end[i]]);
    log_lik[i]= log_mix(rho[person[start[i]]],
              dot_product(RUMlog_prob[start[i]:end[i]],y[start[i]:end[i]]),
              dot_product(RRlog_prob[start[i]:end[i]],y[start[i]:end[i]]));
   }
}

model {  
    RUMdelta ~ normal(0,1);
    RRdelta ~  normal(0,1);
    RUMbeta ~  normal(0,1);
    RRbeta ~   normal(0,1);
    RUMbeta_price ~ normal(0, 1);
    RRbeta_price ~ normal(0, 1);
        RRmu ~ normal(0,1);
    //mu_alpha ~ normal(0,1);
    //alpha ~ normal(0,sigma_alpha);
   // sigma_alpha ~ normal(0, .5);
    gamma ~ normal(0, 0.25);
  // log probabilities of each choice in the dataset
  for(i in 1:T) {
    target += log_lik[i];
  }
  

}

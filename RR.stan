
data {
  int N; // number of rows
  int I; // number of individuals
  int Kx; // number of columns in X
  int T; // number of individual-choice set combination
  vector<lower = 0, upper = 1>[N] y; // choice indicator
  matrix[N,2] trX; // attribute matrix
    matrix[N,2] gfX; // attribute matrix
      matrix[N,2] bdX; // attribute matrix
        matrix[N,2] plX; // attribute matrix
          matrix[N,2] wqX; // attribute matrix
          matrix[N,2] priceX; // attribute matrix          
  int person[N]; // index for individual
  
  int start[T]; // the starting observation for each choice set
  int end[T]; // the ending observation for each choice set
  vector[N] action; // action indicator

}

parameters {
  vector[5] RRbeta;
  real<upper = 0> RRbeta_price;
  real delta;
}

transformed parameters {
  vector[N] linpred; 
  vector[N] log_prob;
  vector[T] log_lik;
  for(idx in 1:N){
  linpred[idx]  =  action[idx]*delta +
            log(1 + exp(trX[idx,1]*RRbeta[1])) + 
            log(1 + exp(trX[idx,2]*RRbeta[1]))+                            
            log(1 + exp(gfX[idx,1]*RRbeta[2])) + 
            log(1 + exp(gfX[idx,2]*RRbeta[2]))+ 
            log(1 + exp(bdX[idx,1]*RRbeta[3])) +
            log(1 + exp(bdX[idx,2]*RRbeta[3]))+ 
            log(1 + exp(plX[idx,1]*RRbeta[4])) + 
            log(1 + exp(plX[idx,2]*RRbeta[4]))+ 
            log(1 + exp(wqX[idx,1]*RRbeta[5])) + 
            log(1 + exp(wqX[idx,2]*RRbeta[5]))+
            log(1 + exp(priceX[idx,1]*RRbeta_price)) + 
            log(1 + exp(priceX[idx,2]*RRbeta_price)) ;
  }
    for (i in 1:T){
    log_prob[start[i]:end[i]] = log_softmax(-linpred[start[i]:end[i]]);
    log_lik[i] = dot_product(log_prob[start[i]:end[i]], y[start[i]:end[i]]);
  }
  
}

model {  
    delta ~ normal(0, 1);
    RRbeta ~  normal(0, 1);
    RRbeta_price ~  normal(0, 1);

  for(i in 1:T) {
    target +=log_lik[i];
  }

}



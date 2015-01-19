return {
  min=100,  -- minimum number of epochs
  max=400,  -- maximum number of epochs
  wrange=0.1, -- weights range initializtion
  wd=0.01,   -- L2 regularization
  mnp=600,     -- max-norm-penalty
  combine="none", -- type of probability combination
  var=0.04,   -- variance of input Gaussian noise perturbation
  actf="relu",  -- activation function
  elman=128,    -- LSTM layer size
  h1=128,      -- first layer size
  h2=128,      -- first layer size
  bsize=256,   -- bunch size (mini-batch)
  dropout=0.0, -- dropout probability
  context=1,   -- context size
  no_channels=true, -- input is not split into channels
  -- random seeds
  wseed=16388,
  sseed=3759,
  pseed=14128,
  pseed2=19488,
  pseed3=4614,
  wnoise=0.0004,
}

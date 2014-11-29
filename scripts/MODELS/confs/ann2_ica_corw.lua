return {
  min=100,  -- minimum number of epochs
  max=400,  -- maximum number of epochs
  wrange=3, -- weights range initializtion
  l1=0.0,   -- L1 regularization
  wd=0.0,   -- L2 regularization
  lr=0.2,   -- learning rate
  mt=0.1,   -- momentum
  decay=0.001, -- learning decay
  mnp=600,     -- max-norm-penalty
  combine="gmean", -- type of probability combination
  var=0.04,   -- variance of input Gaussian noise perturbation
  actf="relu",  -- activation function
  h1=128,      -- first layer size
  h2=128,      -- second layer size
  bsize=128,  -- bunch size (mini-batch)
  dropout=0.5, -- dropout probability
  context=1,   -- context size
  no_channels=true, -- input is not split into channels
  -- random seeds
  wseed=1234,
  sseed=5678,
  pseed=4873,
  pseed2=4948,
  pseed3=9251,
}

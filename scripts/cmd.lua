local cmd = {}

local function add_void(cmd_opt_parser, description, long, short)
  cmd_opt_parser:add_option{
    description = assert(description),
    index_name = assert(long),
    long = long,
    short = short,
    argument = "no",
  }
end

local function add_num(cmd_opt_parser, description, long, short, default, always)
  cmd_opt_parser:add_option{
    description = assert(description),
    index_name = assert(long),
    long = long,
    short = short,
    argument = "yes",
    filter = tonumber,
    default_value = default,
    mode = ((default or always) and "always") or nil,
  }
end

local function add_str(cmd_opt_parser, description, long, short, default, always)
  cmd_opt_parser:add_option{
    description = assert(description),
    index_name = assert(long),
    long = long,
    short = short,
    argument = "yes",
    default_value = default,
    mode = ((default or always) and "always") or nil,
  }
end

function cmd.add_defopt(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  cmd_opt_parser:add_option{
    description = "Default options table",
    index_name = "defopt",
    short = "f",
    argument = "yes",
    filter = function(v)
      return assert(util.deserialize(v), "Impossible to open defopt table")
    end
  }
end

function cmd.add_knn(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_str(cmd_opt_parser, "K value", "KNN", "K", 40)
  add_num(cmd_opt_parser, "Seed", "seed", nil, 1234)
end

function cmd.add_mlp(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_str(cmd_opt_parser, "Activation function", "actf", "a", "tanh")
  add_num(cmd_opt_parser, "Hidden layer 1", "h1", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 2", "h2", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 3", "h3", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 4", "h4", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 5", "h5", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 6", "h6", nil, 0)
  add_num(cmd_opt_parser, "dropout", "dropout", nil, 0.0)
  add_num(cmd_opt_parser, "Learning rate", "lr", "l", 0.01)
  add_num(cmd_opt_parser, "Momentum", "mt", "m", 0.01)
  add_num(cmd_opt_parser, "Weight decay", "wd", "w", 0.01)
  add_num(cmd_opt_parser, "L1 penalty", "l1", nil, 1e-04)
  add_num(cmd_opt_parser, "Decay", "decay", "d", 5e-07)
  add_num(cmd_opt_parser, "Max norm penalty", "mnp", nil, 4)
  add_num(cmd_opt_parser, "Bunch size", "bsize", "b", 384)
  add_num(cmd_opt_parser, "Min epochs", "min", nil, 200)
  add_num(cmd_opt_parser, "Max epochs", "max", nil, 400)
  add_num(cmd_opt_parser, "Weights range", "wrange", nil, 3)
  add_str(cmd_opt_parser, "Loss function", "loss", nil, "cross_entropy")
  add_num(cmd_opt_parser, "Weights seed", "wseed", nil, 1234)
  add_num(cmd_opt_parser, "Shuffle seed", "sseed", nil, 5678)
  add_num(cmd_opt_parser, "Perturbation seed", "pseed", nil, 4873)
  add_num(cmd_opt_parser, "Perturbation seed 2", "pseed2", nil, 4948)
  add_num(cmd_opt_parser, "Perturbation seed 3", "pseed3", nil, 9251)
  add_num(cmd_opt_parser, "Gaussian noise variance", "var", nil, 0.1)
end

function cmd.add_lstm(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_str(cmd_opt_parser, "Activation function", "actf", "a", "relu")
  add_num(cmd_opt_parser, "Hidden layer 0", "h0", nil, 0)
  add_num(cmd_opt_parser, "LSTM layer", "lstm", nil, 32)
  add_num(cmd_opt_parser, "Hidden layer 1", "h1", nil, 32)
  add_num(cmd_opt_parser, "Hidden layer 2", "h2", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 3", "h3", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 4", "h4", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 5", "h5", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 6", "h6", nil, 0)
  add_num(cmd_opt_parser, "dropout", "dropout", nil, 0.0)
  add_num(cmd_opt_parser, "Weight decay", "wd", "w", 0.01)
  add_num(cmd_opt_parser, "Max norm penalty", "mnp", nil, 4)
  add_num(cmd_opt_parser, "Bunch size", "bsize", "b", 1)
  add_num(cmd_opt_parser, "Min epochs", "min", nil, 200)
  add_num(cmd_opt_parser, "Max epochs", "max", nil, 400)
  add_num(cmd_opt_parser, "Weights range", "wrange", nil, 3)
  add_str(cmd_opt_parser, "Loss function", "loss", nil, "cross_entropy")
  add_num(cmd_opt_parser, "Weights seed", "wseed", nil, 1234)
  add_num(cmd_opt_parser, "Shuffle seed", "sseed", nil, 5678)
  add_num(cmd_opt_parser, "Perturbation seed", "pseed", nil, 4873)
  add_num(cmd_opt_parser, "Perturbation seed 2", "pseed2", nil, 4948)
  add_num(cmd_opt_parser, "Perturbation seed 3", "pseed3", nil, 9251)
  add_num(cmd_opt_parser, "Gaussian noise variance", "var", nil, 0.1)
  add_num(cmd_opt_parser, "Weights noise", "wnoise", nil, 0)
end

function cmd.add_elman(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_str(cmd_opt_parser, "Activation function", "actf", "a", "relu")
  add_num(cmd_opt_parser, "Hidden layer 0", "h0", nil, 0)
  add_num(cmd_opt_parser, "Elman layer", "elman", nil, 32)
  add_num(cmd_opt_parser, "Hidden layer 1", "h1", nil, 32)
  add_num(cmd_opt_parser, "Hidden layer 2", "h2", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 3", "h3", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 4", "h4", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 5", "h5", nil, 0)
  add_num(cmd_opt_parser, "Hidden layer 6", "h6", nil, 0)
  add_num(cmd_opt_parser, "dropout", "dropout", nil, 0.0)
  add_num(cmd_opt_parser, "Weight decay", "wd", "w", 0.01)
  add_num(cmd_opt_parser, "Max norm penalty", "mnp", nil, 4)
  add_num(cmd_opt_parser, "Bunch size", "bsize", "b", 1)
  add_num(cmd_opt_parser, "Min epochs", "min", nil, 200)
  add_num(cmd_opt_parser, "Max epochs", "max", nil, 400)
  add_num(cmd_opt_parser, "Weights range", "wrange", nil, 3)
  add_str(cmd_opt_parser, "Loss function", "loss", nil, "cross_entropy")
  add_num(cmd_opt_parser, "Weights seed", "wseed", nil, 1234)
  add_num(cmd_opt_parser, "Shuffle seed", "sseed", nil, 5678)
  add_num(cmd_opt_parser, "Perturbation seed", "pseed", nil, 4873)
  add_num(cmd_opt_parser, "Perturbation seed 2", "pseed2", nil, 4948)
  add_num(cmd_opt_parser, "Perturbation seed 3", "pseed3", nil, 9251)
  add_num(cmd_opt_parser, "Gaussian noise variance", "var", nil, 0.1)
  add_num(cmd_opt_parser, "Weights noise", "wnoise", nil, 0)
end

function cmd.add_cv(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_str(cmd_opt_parser, "Subject name", "subject", nil, nil, true)
  add_void(cmd_opt_parser, "No channels in FFT data", "no_channels")
  add_str(cmd_opt_parser, "FFT data path", "fft", nil, nil, true)
  add_str(cmd_opt_parser, "COR data path", "cor")
  add_str(cmd_opt_parser, "Test output", "test")
  add_str(cmd_opt_parser, "Prefix for validation data output", "prefix", nil, "./")
  add_str(cmd_opt_parser, "Combine algorithm (max, amean, gmean, hmean)",
          "combine", nil, "gmean")
  add_str(cmd_opt_parser, "Save activations output directory",
          "save_activations")
  add_void(cmd_opt_parser, "Add first derivative", "d1")
  add_void(cmd_opt_parser, "Add second derivative", "d2")
  add_num(cmd_opt_parser, "Context size", "context", "c", 0)
end

function cmd.add_help(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  cmd_opt_parser:add_option  {
    description = "Shows this help message",
    short = "h",
    long = "help",
    argument = "no",
    action = function (argument) 
      print(cmd_opt_parser:generate_help()) 
      os.exit(1)
    end    
  }
end

cmd.utils = {
  add_void = add_void,
  add_num  = add_num,
  add_str  = add_str,
}

return cmd

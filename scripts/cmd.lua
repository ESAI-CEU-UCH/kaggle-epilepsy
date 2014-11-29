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
  add_str(cmd_opt_parser, "K value", "KNN", "K", 6)
  add_num(cmd_opt_parser, "Seed", "seed", nil, 1234)
end

function cmd.add_mlp(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_str(cmd_opt_parser, "Activation function", "actf", "a", "tanh")
  add_num(cmd_opt_parser, "Hidden 1", "h1", nil, 256)
  add_num(cmd_opt_parser, "Hidden 2", "h2", nil, 0)
  add_num(cmd_opt_parser, "Hidden 3", "h3", nil, 0)
  add_num(cmd_opt_parser, "Hidden 4", "h4", nil, 0)
  add_num(cmd_opt_parser, "Hidden 5", "h5", nil, 0)
  add_num(cmd_opt_parser, "Hidden 6", "h6", nil, 0)
  add_void(cmd_opt_parser, "Pretrain", "pretrain")
  add_num(cmd_opt_parser, "dropout", "dropout", nil, 0.0)
end

function cmd.add_cnn(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_str(cmd_opt_parser, "Activation function", "actf", "a", "tanh")
  add_num(cmd_opt_parser, "Convolution 1", "c1", nil, 10)
  add_num(cmd_opt_parser, "Convolution 2", "c2", nil, 20)
  add_num(cmd_opt_parser, "Hidden 1", "h1", nil, 256)
  add_num(cmd_opt_parser, "Hidden 2", "h2", nil, 0)
end

function cmd.add_sgd(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_num(cmd_opt_parser, "Learning rate", "lr", "l", 0.01)
  add_num(cmd_opt_parser, "Momentum", "mt", "m", 0.01)
  add_num(cmd_opt_parser, "Weight decay", "wd", "w", 0.01)
  add_num(cmd_opt_parser, "L1 penalty", "l1", nil, 1e-04)
  add_num(cmd_opt_parser, "Decay", "decay", "d", 5e-07)
  add_num(cmd_opt_parser, "Max norm penalty", "mnp", nil, 4)
  add_void(cmd_opt_parser, "Only reg. last layer", "reg_last_layer")
end

function cmd.add_cg(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_num(cmd_opt_parser, "max-iter", "max_iter")
  add_num(cmd_opt_parser, "Weight decay", "wd", "w", 0.01)
  add_num(cmd_opt_parser, "L1 penalty", "l1", nil, 1e-04)
  add_num(cmd_opt_parser, "Max norm penalty", "mnp", nil, 4)
  add_void(cmd_opt_parser, "Only reg. last layer", "reg_last_layer")
end

function cmd.add_EM(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
end

function cmd.add_supervised(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_num(cmd_opt_parser, "Bunch size", "bsize", "b", 384)
  add_num(cmd_opt_parser, "Min epochs", "min", nil, 200)
  add_num(cmd_opt_parser, "Max epochs", "max", nil, 400)
  add_num(cmd_opt_parser, "Weights range", "wrange", nil, 3)
  add_str(cmd_opt_parser, "Loss function", "loss", nil, "cross_entropy")
end

function cmd.add_seeds(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_num(cmd_opt_parser, "Weights seed", "wseed", nil, 1234)
  add_num(cmd_opt_parser, "Shuffle seed", "sseed", nil, 5678)
  add_num(cmd_opt_parser, "Perturbation seed", "pseed", nil, 4873)
  add_num(cmd_opt_parser, "Perturbation seed 2", "pseed2", nil, 4948)
  add_num(cmd_opt_parser, "Perturbation seed 3", "pseed3", nil, 9251)
end

function cmd.add_cv(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_num(cmd_opt_parser, "Number of channels", "channels", "n", 16)
  add_void(cmd_opt_parser, "Ignore channels", "no_channels")
  add_void(cmd_opt_parser, "Mean channel", "mean_channel")
  add_str(cmd_opt_parser, "Train list", "list", nil, 
          "/home/experimentos/KAGGLE/EPILEPSY_PREDICTION/lists/Dog_1_FFT_60s_30s_BFPLOS.train")
  add_str(cmd_opt_parser, "Train correlations", "cor")
  add_str(cmd_opt_parser, "Test output", "test")
  add_str(cmd_opt_parser, "Append validation result", "append")
  add_void(cmd_opt_parser, "Postprocess threshold", "th")
  add_void(cmd_opt_parser, "ZCA whitening", "zca")
  add_void(cmd_opt_parser, "PCA dimensionality reduction", "pca")
  add_void(cmd_opt_parser, "PCA/ZCA by rows", "by_rows")
  add_void(cmd_opt_parser, "Ignore 'bad' models", "ignore_bad_models")
  add_num(cmd_opt_parser, "Resample validation", "sampling", nil, 0)
  add_void(cmd_opt_parser, "CV adaptation", "cv_adaptation")
  add_str(cmd_opt_parser, "Prefix for validation data output", "prefix")
  add_str(cmd_opt_parser, "Combine validation result (max, amean, gmean, hmean)",
          "combine", nil, "max")
  add_str(cmd_opt_parser, "Save activations output directory",
          "save_activations")
  add_void(cmd_opt_parser, "Add first derivative", "d1")
  add_void(cmd_opt_parser, "Add second derivative", "d2")
  add_num(cmd_opt_parser, "Context size", "context", "c", 0)
end

function cmd.add_pca_zca(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_num(cmd_opt_parser, "Number of channels", "channels", "n", 16)
  add_void(cmd_opt_parser, "Mean channel", "mean_channel")
  add_str(cmd_opt_parser, "Train list", "list", nil, 
          "/home/experimentos/KAGGLE/EPILEPSY_PREDICTION/lists/Dog_1_FFT_60s_30s_BFPLOS.train")
  add_str(cmd_opt_parser, "Destination directory", "dest", "d", nil, true)
  add_void(cmd_opt_parser, "ZCA whitening", "zca")
  add_void(cmd_opt_parser, "PCA dimensionality reduction", "pca")
  add_void(cmd_opt_parser, "PCA/ZCA by rows", "by_rows")
end

function cmd.add_noise(cmd_opt_parser)
  assert(cmd_opt_parser, "Needs a cmdOpt object")
  add_num(cmd_opt_parser, "Gaussian noise variance", "var", nil, 0.1)
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

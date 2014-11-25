--[[
  This file is part of ESAI-CEU-UCH/kaggle-epilepsy (https://github.com/ESAI-CEU-UCH/kaggle-epilepsy)
  
  Copyright (c) 2014, ESAI, Universidad CEU Cardenal Herrera,
  (F. Zamora-Martínez, F. Muñoz-Malmaraz, P. Botella-Rocamora, J. Pardo)
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
]]
local FFT_DATA_PATH = assert(arg[1], "Needs 1st argument with FFT source path")
local PCA_DATA_PATH = assert(arg[2], "Needs 2nd argument with PCA source path")
local OUTPUT_PATH = assert(arg[3], "Needs 3rd argument with destination path")

package.path = package.path .. ";./scripts/?.lua"
--
local common = require "common"

local read = matrix.fromTabFilename

local pca_data = {}
for subject in ipars{ "Dog_1", "Dog_2", "Dog_3", "Dog_4", "Dog_5",
                      "Patient_1", "Patient_2" } do
  local center = read("%s/%s_pca_center.txt"%{PCA_DATA_PATH, subject})
  local scale = 1/read("%s/%s_pca_scale.txt"%{PCA_DATA_PATH, subject})
  local rotation = read("%s/%s_pca_rotation.txt"%{PCA_DATA_PATH, subject})
  center = center:rewrap(center:size())
  scale = scale:rewrap(scale:size())
  --
  local transform = function(x)
    for _,x_row in matrix.ext.iterate(x, 1) do
      x_row[{}] = (x_row - center):cmul(scale)
    end
    return x * rotation
  end
  --
  local files_iterator = iterator(io.popen("ls %s/%s*channel_01*"%{FFT_PCA_PATH,
                                                                   subject}):lines())
  for filename in files_iterator do
    local mask = filename:gsub("channel_01", "channel_??")
    local m = matrix.join(2, iterator(glob(mask)):map(read):table())
    local out = transform(m)
    out:toTabFilename("%s/%s.txt"%{OUTPUT_PATH,
                                   filename:gsub(".channel_.*$","")})
  end
end

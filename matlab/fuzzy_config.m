function cfg = fuzzy_config()
%FUZZY_CONFIG 读取仓库共享的唯一配置源 shared/fuzzy_config.json
%
%   Python、MATLAB 与自动生成的 fuzzy_pid.fis 全部由这一份 JSON 驱动，
%   从机制上消除"同一套算法在不同语言下规则库不一致"的问题。
%
%   用法:
%       cfg = fuzzy_config();
%       cfg.fuzzy.rules.dkp        % 49 条 ΔKp 规则 (7x7, 取值 -3..3)
%       cfg.scenarios.thermal.pid  % 场景一控制器参数
%
%   注意: 结果做了 persistent 缓存，脚本运行期间只读取一次文件。

persistent CACHE
if ~isempty(CACHE)
    cfg = CACHE;
    return;
end

here = fileparts(mfilename('fullpath'));
cands = { fullfile(here, 'shared', 'fuzzy_config.json'), ...
          fullfile(fileparts(here), 'shared', 'fuzzy_config.json'), ...
          fullfile(fileparts(fileparts(here)), 'shared', 'fuzzy_config.json') };

path = '';
for k = 1:numel(cands)
    if isfile(cands{k})
        path = cands{k};
        break;
    end
end

if isempty(path)
    error('fuzzy_config:missing', ...
        ['未找到 shared/fuzzy_config.json，请保持仓库目录结构完整。\n已尝试:\n  %s'], ...
        strjoin(cands, '\n  '));
end

cfg = jsondecode(fileread(path));
cfg.source_path = path;
CACHE = cfg;
end

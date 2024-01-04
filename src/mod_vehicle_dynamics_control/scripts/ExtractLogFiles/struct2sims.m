function s_out = struct2sims(s_in, name)
    assert(isscalar(s_in) && ~isempty(s_in), ...
        'only non-empty scalar structures are supported');
 
    if nargin < 2, name = inputname(1); end
 
    s_out = struct();
    f = fieldnames(s_in);
    for ii = 1:numel(f)
        if(isempty(name))
          subname = f{ii};
        else
          subname = [name '_' f{ii}];
        end
        val = s_in.(f{ii});
        if isstruct(val)
            % recursively flatten struct, then unpack its fields
            s_tmp = struct2sims(val, subname);
            ff = fieldnames(s_tmp);
            for jj = 1:numel(ff)
                s_out.(ff{jj}) = s_tmp.(ff{jj});
            end
        else
            % assign non-struct fields directly
            s_out.(subname) = val;
        end
    end
end
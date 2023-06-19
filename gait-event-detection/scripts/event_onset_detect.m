function id_onset = event_onset_detect(acc, fs)

    % 0. Generate time data
    time = (1:length(acc))';
    
    % 1. Get acceleration norm
    acc_norm = sqrt( ...
        sum(acc.^2, 2) ...
        );
    
    % 2. Filter acceleration
    filter_freq = meanfreq(acc_norm, fs);
    norm_freq = filter_freq / (fs/2);
    [b_filt, a_filt] = butter(2, norm_freq);
    acc_norm_filt = filtfilt(b_filt, a_filt, acc_norm);
    
    % 3. Get peak projection point
    [~,id] = max(acc_norm_filt);
    % Get vector connecting signal start to peak projection point
    vector = [time(1),acc_norm_filt(1); time(id),acc_norm_filt(id)];
    
    % 4. Get orthogonal projections from start to projection
    warning('off')
    for i=1:id
        % Get query point
        query_point = [time(i),acc_norm_filt(i)];
        % Get vector endpoints
        p0 = vector(1,:);
        p1 = vector(2,:);
        % Get projection from query point to vector
        a = [-query_point(1)*(p1(1)-p0(1)) - query_point(2)*(p1(2)-p0(2));
            -p0(2)*(p1(1)-p0(1)) + p0(1)*(p1(2)-p0(2))]; 
        b = [p1(1) - p0(1), p1(2) - p0(2);
            p0(2) - p1(2), p1(1) - p0(1)];
        proj_point = -(b\a);
        % Check projection point orientation
        if diff([query_point(2) proj_point(2)]) > 0
            % Get projection vector size
            proj_sz(i,1) = sqrt( ...
                diff([query_point(1) proj_point(1)]).^2 ...
                + diff([query_point(2) proj_point(2)]).^2 ...
                );
        else
            proj_sz(i,1) = 0;
        end
    end
    
    % 5. Get timepoint of longest projection vector
    [~, id_onset] = max(proj_sz);

end

function values = generate_samples(range, n_samples, min_separation)
    % GENERATE_SAMPLES Generate two random values per sample from a given range
    % ensuring the values are not too close together by forcing separation.
    %
    % Inputs:
    %   range         - [min, max] range for random values
    %   n_samples     - Number of samples to generate
    %   min_separation - Minimum allowable difference between the two values
    %
    % Output:
    %   values        - n_samples x 2 matrix of generated values

    values = zeros(n_samples, 2);

    % Extract range boundaries
    r_min = range(1);
    r_max = range(2);

    % Ensure the range is valid
    if abs(r_max - r_min) < min_separation
        error('Range is too small for the specified minimum separation.');
    end

    for i = 1:n_samples
        % Generate v1 in the range [r_min, r_max - min_separation)
        v1 = r_min + (r_max - r_min - min_separation) * rand();

        % Generate v2 in the range (v1 + min_separation, r_max]
        v2 = v1 + min_separation + (r_max - (v1 + min_separation)) * rand();

        % Store the values as a row [v1, v2]
        values(i, :) = [v1, v2];
    end
end

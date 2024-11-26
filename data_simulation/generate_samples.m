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

    % Extract range boundaries
    r_min = range(1);
    r_max = range(2);

    % Ensure the range is valid
    if abs(r_max - r_min) < min_separation
        error('Range is too small for the specified minimum separation.');
    end

    % v1 range and rand value
    range1 = r_max - r_min - min_separation;
    value1 = r_min + range1 * rand(n_samples, 1);

    % v2 range and rand value
    range2 = r_max - value1;
    value2 = value1 + range2 * rand(n_samples, 1);

    % Combine values into a 2-column matrix
    values = [value1, value2];
end

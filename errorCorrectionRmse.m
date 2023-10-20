function [ Weights, iterations ] = errorCorrectionRmse ( Input, eps, Start, W )
% errorCorrection(Input) returns a weighting vector for the input/output
% mapping presented by N x (n+1) matrix Input containing N rows of learning
% samples, n inputs each foollowed by a desired output, found using the
% error-correction learning rule
% If Start = 1, then starting weights are read from W, otherwise random
% weights should be generated
% eps is a tolerance threshold to stop learning process

%Create a dynamic Array
rmse_values = [];

% determine size of the Input matrix
[N, m] = size(Input);
% N input samples and m-1 inputs

% vector of the length m for storing weights
Weights = zeros(1, m);

if (Start == 1)
    if m ~= length(W)
        disp('Starting weighting vector and the number of inputs do not match');
    else
        Weights = W;
    end
else
    % randomization of the random numbers generator based on the current
    % time
    rng('shuffle');
    % Starting weights are generated as random nubbers from [-0.5, 0.5]
    Weights = rand(1, m) - 0.5;
    
end

% Shifting of all columns of Input by 1 to the right to release the 1st
% column
Input(:,2:m+1) = Input(:, 1:m);

% Generation of the column of ones
X0 = ones(N, 1);

% Appending a column of ones as the 1st column to Input
Input(:, 1) = X0;

% Calculation of reciprocal inputs (last column of Input (m+1) contains
% desired outputs and should not be targeted)
%Input_1(:, 1 : m) = Input(:, 1 : m).^-1;

% f - a vector of desired outputs
f = Input(:,m+1);
% Input will contain now only inputs, without desired outputs
Input = Input(:, 1:m);

%learningRate is equal to 1/(number of weights)
learningRate = 1/(m);

% learning is a flag (true - a neuron has to learn; false - learning
% finished
learning = true;
%counter of iterations
iterations = 0;

% Mapping a k-valued discrete I/O represented by integers.
subinterval_width = 2/m;
mapped_input = -1 + (Input + 0.5) * subinterval_width;
mapped_f = -1 + (f + 0.5)* subinterval_width;

mapped_input_1(:, 1 : m) = mapped_input(:, 1 : m).^-1;

eps = 1 / (m+100);

while (learning == true)
    % flipping a flag at the beginning of every iteration
    learning = false;
    iterations = iterations + 1;
    
    % weighted sums for all learning samples, Input is taken transposed to
    % calculate all of them simultaneously (each row of input is a sample)
    Z = Weights * mapped_input';
    % actual outputs for all samples
    Y = tanh(Z);
    %MSE
    MSE = sum((mapped_f - Y').^2)/N;
    %RMSE
    RMSE = sqrt(MSE);
    rmse_values = [rmse_values, RMSE];
    
    if (RMSE <= eps)
        %Plot the RMSE values
        plot(1:iterations, rmse_values);
        xlabel('Iterations');
        ylabel('RMSE');
        title('RMSE and Iterations');
       
        disp('Output before conversion:'); disp(Y);

        for i = 1:length(Y)
        if (Y(i) >= -1 && Y(i) < -0.5)
            Y(i) = 0;
        elseif all(Y(i) > -0.5 && Y(i) < 0)
            Y(i) = 1;
        elseif all(Y(i) > 0 && Y(i) < 0.5)
            Y(i) = 2;
        else
            Y(i) = 3;
        end
        end

        disp('Output after conversion:'); disp(Y);
        break  % exit a loop

    else
        learning = true; % has to learn
        end

    % A loop over all learning samples
    for j = 1 : N
        % z = w0 + w1*x1 +...+ wn1*xn - weighted sum
        z = dot(mapped_input(j, :), Weights);
        % actual output = activation function of the weighted sum
        y = tanh(z);
        if (y ~= mapped_f(j))  % if actual output ~= desired output
            learning = true; % then flip learning flag 
            error = mapped_f(j) - y; % calculate the error
            % adjust the weights
            Weights = Weights + learningRate * error * mapped_input_1(j, :)/m;
        end
    end
end

end


load('data.csv');

% Plot positions
figure;

plot(data(:,[1,4,7]),data(:,[2,5,8]));
legend({'True Position', 'Prediction only','EKF Estimate'});
xlabel('x');
ylabel('y');
title('Position Estimate');

figure;
indices = 1:1:size(data, 1);
indices = indices';
plot(indices, data(:,3), indices, data(:, 6), indices, data(:, 9));
legend({'True Position', 'Prediction only', 'EKF Estimate'});
xlabel('Iteration');
ylabel('Radian');
title('Orientation Estimate');

% Estimation error (euclidian distance)
figure;
title('Estimate errors from true position (Euclidian distance)');
ekf_error = sqrt((data(:,1)-data(:,7)).^2 + (data(:,2)-data(:,8)).^2);
ekf_error_ori = abs(data(:,3)-data(:,9));
plot(indices, ekf_error, indices, ekf_error_ori);
legend({'EKF Translation Error', 'EKF Rotation Error'});
xlabel('Iteration')
ylabel('Error')

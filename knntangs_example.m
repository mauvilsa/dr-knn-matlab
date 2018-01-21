% read the data
load('data.mat');

% split data in training and test
Xtr = data;
Xtr_id = data_id;
Xte = data(:,1:3:end);
Xte_id = data_id(1:3:end);
Xtr(:,1:3:end) = [];
Xtr_id(1:3:end) = [];

% compute tangent vectors (hvrs: horizontal and vertical translations, rotation and scaling)
Vtr = tangVects( Xtr, 'hvrs' );
Vte = tangVects( Xte, 'hvrs' );

% estimage knn error
knnerr_euclidean = classify_knn( Xtr, Xtr_id, 1, Xte, Xte_id );
knnerr_rtangent = classify_knn( Xtr, Xtr_id, 1, Xte, Xte_id, 'rtangent', 'tangVp',Vtr );
knnerr_otangent = classify_knn( Xtr, Xtr_id, 1, Xte, Xte_id, 'otangent', 'tangVx',Vte );
knnerr_atangent = classify_knn( Xtr, Xtr_id, 1, Xte, Xte_id, 'atangent', 'tangVp',Vtr, 'tangVx',Vte );
knnerr_tangent = classify_knn( Xtr, Xtr_id, 1, Xte, Xte_id, 'tangent', 'tangVp',Vtr, 'tangVx',Vte );

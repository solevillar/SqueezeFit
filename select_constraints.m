%Given data:dxN, (d dimension, N number of data points), and labels:N, for
%each point in the data set if finds the k nearest neighbors with each of
%the different labels from it, and computes the corresponding constraint
%(Pi-Pj). Delta:dxm is the set of constraints to be used in the SqueezeFit
%SDP. smallest is a real postive number corresponding to the smallest norm 
%of vectors in Delta. If k=0 it computes all pairwise constraints (not 
%recommendable since it'd be very inefficient).
function [Delta, smallest]= select_constraints(data, labels, k)

samples=double(data');
[n_total, d] = size(samples);
class_num = length(unique(labels));

% separate samples into different arrays based on label
separated = cell(1, class_num);
for i=1:n_total
    separated{labels(i)+1} = cat(1, separated{labels(i)+1}, samples(i,:));
end


idx=1;
smallest=intmax;

if k==0
 Delta=zeros(d,0);
    %all constraints
 for i=1:n_total
     for j=i+1:n_total
         if labels(i)~=labels(j)
             delta=data(:,i)-data(:,j);
             Delta(:,idx)=delta;
             if smallest>norm(delta)^2
                 smallest=norm(delta)^2;
             end
             idx=idx+1;
         end
     end
 end
else 
Delta=zeros(d,k*n_total);

for c=1:class_num
    % Separate data by current and other classes
    current_class = separated{c};
    other_classes = cat(1, separated{1:end ~= c});
    
    % Search for kNN
    nn_idxs = knnsearch(other_classes, current_class, 'k', k);
    
    for i=1:size(current_class,1)
        % Loop over number of nearest neighbors
        for j=1:k
            % Get current nearest neighbor index in nn_idxs
            nn_idx = nn_idxs(i, j);
            delta = current_class(i,:) - other_classes(nn_idx, :);
            if smallest>norm(delta)^2
                 smallest=norm(delta)^2;
             end
            Delta(:,idx)=double(delta');   
            idx=idx+1;
        end
    end
end


end
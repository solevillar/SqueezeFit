function [data,labels,test_data,test_labels]= load_mnist(file_data, file_labels, file_test, file_test_labels)    
    tic;
    data_T = loadMNISTImages(file_data);
    labels_T = loadMNISTLabels(file_labels);
    test_T = loadMNISTImages(file_test);
    test_labels_T = loadMNISTLabels(file_test_labels);
    toc
    
    nines_idx=find(labels_T==9);
    fours_idx=find(labels_T==4);
    nines_test_idx=find(test_labels_T==9);
    fours_test_idx=find(test_labels_T==4);
    
    data=data_T(:, vertcat(nines_idx,fours_idx));
    labels=labels_T(vertcat(nines_idx,fours_idx));
    
    test_data= test_T(:, vertcat(nines_test_idx,fours_test_idx));
    test_labels= test_labels_T(vertcat(nines_test_idx,fours_test_idx));
    
    %manually look for labels:
    for i=1:size(labels,1)
        if labels(i)==4
            labels(i)=0;
        else
            labels(i)=1;
        end
    end
    for i=1:size(test_labels,1)
        if test_labels(i)==4
            test_labels(i)=0;
        else
            test_labels(i)=1;
        end
    end
end
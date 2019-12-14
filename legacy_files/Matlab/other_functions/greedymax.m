function bins = greedymax(K,k,p)

    n = size(K,1);% e = sum(sum(A))/2;

%initialise
    Uv = 1:n; %unexplored vertices
%     Ue = K~=1; %unexplored edges
%     i = 1; %current vertex 
    j = 1; %current partition
    Ee = zeros(size(K)); %explored edges 
%     KMean = Inf;
    bins = zeros(1,n);
    
    while ~isempty(Uv) == 1 %stop if all vertices explored
        
        %start at random vertex
        rn = floor(rand*(length(Uv)))+1;               
        Ev = Uv(rn); 
        
        cond = 1; i = 1; 
        while cond %stop if cannot find vertex with m>k*mcurv      
    
            %get all unexplored incident edges to Ev(i)
            vj = find(K(Ev(i),:));
        
            if ~isempty(vj) == 1 
            
                %explore edge eij with highest curvature
                [KMax, ind] = max(K(Ev(i),vj)); 
                vj = vj(ind);
                Ee(Ev(i),vj) = 1; %make eij explored 
                Ee(vj,Ev(i)) = 1;
            
                %make eij explored if it has higher than k*average curv
                KMean = mean(K(Ee>0)); %mean curvature of partition 
                if KMax >= KMean%abs((KMax - KMean)/KMean) < k  
                    K(Ev(i),vj) = 0; %remove eij from unexplored   
                    K(vj,Ev(i)) = 0;
                    Uv(Uv==vj) = []; %remove vj from unexplored     
                    Ev(i+1) = vj; %make vj explored    
                    bins(vj) = j;
                    i = i+1; 
                    nnz(bins)
                    
                    %% delete afterwards
ColOrd = get(gca,'ColorOrder'); m = size(ColOrd,1);
for l = 1:size(K,1)
    ColRow = rem(bins(l),m);
    if ColRow == 0
        ColRow = m;
    end
    Col = ColOrd(ColRow,:);
    highlight(p,l,'NodeColor',Col)
end
                    
                else %backtrack  
                    disp('backtrack - lower than average')
                    Ee(Ev(i),vj) = 1; %remove from explored
                    Ev = Ev(1:i-1);
                    i = i-1;
                end
            
            else %backtrack
                disp('backtrack - no more neighbours')
                Ev = Ev(1:i);
                i = i-1;
            end   
        
            %break if no more vertex possible    
            if i == 0
                cond = 0;
%                 bins(Ev) = j;
%                 Uv = setdiff(Uv,Ev); 
                Ev = [];
                Ee = zeros(size(K));         
                j = j+1;                 
            end
        end
    end
function [V_tot,tau_vals,ε_mins,gammas] = pecora_embed(varargin)
% PECORA_EMBED is a unified approach to properly embed a time series based
% on the paper of Pecora et al., Chaos 17 (2007).
%
% Minimum input-arguments: 1
% Maximum input-arguments: 7
%
% [V, taus, ε_mins, gammas] = pecora_embed(s,tau_max,...
%                                   ε_tries,datasample,...
%                                   theiler_window,norm,break_percentage);
%
% This function embeds the input time series 's' with different delay
% times tau. The approach views the problem of choosing all embedding
% parameters as being one and the same problem addressable using a single
% statistical test formulated directly from the reconstruction theorems.
% This allows for varying time delays appropriate to the data and
% simultaneously helps decide on embedding dimension. A second new
% statistic, undersampling, acts as a check against overly long time delays
% and overly large embedding dimension.
%
% Input:
%
% 's'               A uni- or multivariate time series, which needs to be
%                   embedded. If the input data is a multivariate set, the
%                   algorithm scans all time series and chooses the time
%                   delays and time series for the reconstruction
%                   automatically.
% 'tau_max'         Defines up to which maximum delay time tau the
%                   algorithm shall look (Default is tau = 50).
% 'epsilion_tries'  Defines how many ε refinements are made for the
%                   computation of the continuity stastistic. Specifically
%                   the range of the input time series is divided by this
%                   input number in order to achieve the ε
%                   refinements the algorithms tests (Default is 20).
% 'datasample'      Defines the size of the random phase space vector
%                   sample the algorithm considers for each tau value, in
%                   order to compute the continuity statistic. This is a
%                   float from the intervall (0 1]. The size of the
%                   considered sample is 'datasample'*length of the current
%                   phase space trajectory (Default is 0.5, i.e. half of
%                   the trajectory points will be considered).
% 'theiler_window'  Defines a temporal correlation window for which no
%                   nearest neighbours are considered to be true, since
%                   they could be direct predecessors or successors of the
%                   fiducial point (Default is 1).
% 'norm'            norm for distance calculation in phasespace. Set to
%                   'euc' (euclidic) or 'max' (maximum). Default is Maximum
%                   norm.
% 'break_percentage'is the fraction of the standard deviation of the
%                   continuity statistic (of the first cycle of embedding),
%                   for which the algorithm breaks the computations.
%
%
%
% Output: (minimum 1, maximum 4)
%
% 'V'               The embedded phase space trajectory
% 'taus'            The (different) time delays chosen by the algorithm
% 'ε_mins'    Continuity statistic. A cell array storing all ε-
%                   mins as a function of 'taus' for each encountered
%                   dimension, i.e. the size of 'ε_mins' is the same
%                   as the final embedding dimension.
% 'gammas'          Undersampling statistic. A cell array storing all gamma
%                   values as a function of 'taus' for each encountered
%                   dimension, i.e. the size of 'gammas' is the same
%                   as the embedding final dimension.


% Copyright (c) 2019
% K. Hauke Kraemer,
% Potsdam Institute for Climate Impact Research, Germany
% http://www.pik-potsdam.de
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or any later version.

%% Assign input

% the input time series. Unlike the description in the docstring, yet there
% is just a univariate time series allowed
s = varargin{1};
% normalize time series
s = (s-mean(s))/std(s);

% make the input time series a column vector
if size(s,1)>size(s,2)
    s = s';
end
% as mentioned above, this will be changed in the end for allowing
% multivariate input
if size(s,1)~=1
    error('only univariate time series allowed for input')
end

try
    tau_max = varargin{2};
catch
    tau_max = 50;
end


try
    eps_tries = varargin{3};
catch
    eps_tries = 20;
end

try
    sample_size = varargin{4};
    if sample_size < 0 || sample_size > 1
        warning('break percentage input must be a value in the interval [0 1]')
        sample_size = 0.5;
    end
catch
    sample_size = 0.5;
end

try
    theiler = varargin{5};
catch
    theiler = 1;
end


methLib={'euc','max'}; % the possible norms
try
    norm = varargin{6};
    if ~isa(norm,'char') || ~ismember(norm,methLib)
       warning(['Specified norm should be one of the following possible values:',...
           10,sprintf('''%s'' ',methLib{:})])
       norm = 'max';
    end
catch
    norm = 'max';
end

try
    break_percentage = varargin{7};
    if break_percentage < 0 || break_percentage > 1
        warning('break percentage input must be a value in the interval [0 1]')
        break_percentage = 0.1;
    end

catch
    break_percentage = 0.1;
end


% set at which fraction the ε is supposed to start (e.g.
% rangefactor = 2 means the first ε value is chosen to be 1/2 of the
% range of the input time series.)  I incorporated this in order to save
% computation time and "focus" on smaller scales for the ε statistic.
% This can be easily omitted in final implementation.
rangefactor = 1;

% confidence level for undersampling statistic (could also be input
% parameter in final implementation)
β = 0.05;

% Matlab in- and output check
narginchk(1,7)
nargoutchk(1,4)

%% Start computation

% intial phase space vector (no embedding)
V_old = s;

% set a flag, in order to tell the while loop when to stop. Each loop
% stands for encountering a new embedding dimension
flag = true;

% set index-counter for the while loop
cnt = 1;

% initial tau value for no embedding. This is trivial 0, when there is no
% embedding
tau_vals = 0;

% loop over increasing embedding dimensions until some break criterion will
% tell the loop to stop/break
while flag

    % preallocate storing vector for minimum ε vals (continuity
    % statistic)
    ε_min_avrg = zeros(1,tau_max+1);

    % preallocate storing vector for gamma vals (undersampling statistic)
    Γ = zeros(1,tau_max+1);

    % set index-counter for the upcoming for-loop over the different tau
    % values
    tau_counter = 1;
    % loop over the different tau values. Starting at 0 is important,
    % especially when considering a mutlivariate input data set to choose
    % from (in the final implemenation)
    for tau = 0:tau_max

        % create new phase space vector. 'embed2' is a helper function,
        % which you find at the end of the script. At this point one could
        % incorporate onther loop over all the input time series, when
        % allowing multivariate input.
        V_new = embed2(V_old,s,tau);

        % select a random phase space vector sample. One could of course
        % take all points from the actual trajectory here, but this would
        % be computationally overwhelming. This is why I incorporated the
        % input parameter 'sample_size' as the fraction of all phase space
        % points.
        data_samps = datasample(1:size(V_new,1),floor(sample_size*size(V_new,1)),...
            'Replace',false);

        % preallocate storing vector for minimum ε from the
        % continuity statistic
        ε_star = zeros(1,floor(sample_size*size(V_new,1)));

        % preallocate storing vector for maximum gamma from the
        % undersampling statisctic
        gamma_k = zeros(1,floor(sample_size*size(V_new,1)));

        % loop over all fiducial points, look at their neighbourhoods and
        % compute the continuity as well as the undersampling statistic for
        % each of them. Eventually we average over all the values to get
        % the continuity and undersampling statistic for the actual tau
        % value.
        for k = 1:floor(sample_size*size(V_new,1))

            % bind the fiducial point from the trajectory sample
            fiducial_point = data_samps(k);

            % compute distances to all other points in dimension d. Vou'll
            % find the helper function at the end of the script
            distances = all_distances(V_new(fiducial_point,1:end-1),...
                                                    V_new(:,1:end-1),norm);

            % compute distances to all other points in dimension d+1 and
            % also the componentwise distances for the undersampling
            % statistic
            [distances2,comp_dist] = all_distances(V_new(fiducial_point,:),...
                                                            V_new,norm);

            % sort these distances in ascending order
            [~,ind] = sort(distances);
            [~,ind2] = sort(distances2);

            % 1) perform undersampling statistic test. Vou'll find the
            % helper function undersampling at the end of this script.

            % herefore get the componentwise distances to compare against
            dist = comp_dist(ind2(2),:);
            % now run the undersampling statistic function on the first
            % component of the phase space vector and the "new" last
            % component, since the first component is the unshifted time
            % series. In case of a multivariate input signal this needs to
            % be done for all components, since they could possibly
            % originate from different input time series.
            [bo,gamma] = undersampling(V_new(:,1),V_new(:,end),dist,β);

            % take the maximum gamma and the corresponding logical
            [gamma_k(k),ind3] = max(gamma);
            bool = bo(ind3);

            % 2) compute the continuity statistic

            % generate the possible εs
            % here the not really necessary 'rangefactor' I mentioned above
            % comes into play, in order to "focus" on the interesting,
            % smaller scales. As I said, this can just be omitted and does
            % not really change anything, unless 'eps_tries' is large
            % enough in order to provide a decent resolution
            εs = linspace(range(s)/rangefactor,0,eps_tries+1);

            % loop over all δ-neighbourhoods. Here we take the table
            % from the paper, which is based on an confidence level β =
            % 0.05. In the final implementation one should be able to
            % choose an β, maybe at least from 0.05 or 0.01. Therefore
            % one could look up a similar table (binomial distribution). We
            % try all these number of points, i.e. we try many different
            % δ's, as mentioned in the paper. For each of the δ
            % neighbourhoods we loop over the εs (decreasing values)
            % and stop, when we can not reject the null anymore. After
            % trying all δs we pick the maximum ε from all
            % δ-trials. This is then the final ε for one specific
            % tau and one specific fiducial point. Afterwards we average
            % over all fiducial points.

            % table from the paper corresponding to β = 0.05
            δ_points = [5 6 7 8 9 10 11 12 13];
            ε_points = [5 6 7 7 8 9 9 9 10];

            % preallocate storing vector for the εs from which
            % we cannot reject the null anymore (for each δ)
            ε_star_δ = zeros(1,length(δ_points));

            % loop over all the δs (from the table above)
            for δ = 1:length(δ_points)

                neighbours = δ_points(δ);
                neighbour_min = ε_points(δ);

                % loop over an decresing ε neighbourhood
                for epsi = 1:length(εs)-1

                    % bind the actual neighbourhood-size
                    epsil = εs(epsi);

                    % define upper and lower ε neighbourhood bound,
                    % that is, the last component of the "new" embedding
                    % vector +- the ε neighbourhood. See Figure 1 in
                    % the paper
                    upper = V_new(fiducial_point,size(V_new,2))+epsil;
                    lower = V_new(fiducial_point,size(V_new,2))-epsil;

                    % scan neighbourhood of fiducial point and count the
                    % projections of these neighbours, which fall into the
                    % ε set

                    count = 0;
                    l = 2; % start with the first neighbour which is not
                           % the fiducial point itself

                    % loop over all neighbours (determined by the δ-
                    % neighbourhood-size) and count how many of those
                    % points fall within the ε neighbourhood.
                    for nei = 1:neighbours

                        % this while loop gurantees, that we look at a true
                        % neighbour and not a one which lies in the
                        % correlation window of the fiducial point
                        while true

                            % check whether the last component of this
                            % neighbour falls into the ε set.
                            % Therefore, first check that the neighbour is
                            % not in the temporal correlation window
                            % (determined by the input 'theiler').
                            % If it is, go on to the next.
                            if ind(l) > fiducial_point + theiler || ...
                                    ind(l) < fiducial_point - theiler

                                % check if the valid neighbour falls in the
                                % ε neighbourhood. If it does, count
                                % it
                                if V_new(ind(l),size(V_new,2))<=upper &&...
                                        V_new(ind(l),size(V_new,2)) >=lower
                                    count = count + 1;
                                end
                                % go to the next neighbour
                                l = l + 1;
                                break
                            else
                                % check the next neighbour
                                l = l + 1;
                            end
                            % make sure the data set is sufficiently
                            % sampled, if not pass an error. Since 'ind' is
                            % a vector storing all indices of nearest
                            % neighbours and l is exceeding its length
                            % would mean that all neighbours are so close
                            % in time that they fall within the correlation
                            % window OR the number of neighbours one is
                            % looking at (see table 1 in the paper) is too
                            % large
                            if l > length(ind)
                                error('not enough neighbours')
                            end
                        end

                    end

                    % if the number of neighbours from the δ
                    % neighbourhood, which get projected into the ε
                    % neighbourhood (Fig.1 in the paper) are smaller than
                    % the amount needed to reject the null, we break and
                    % store this particular ε value (which
                    % determines the ε neighbourhood-size)
                    if count < neighbour_min
                        ε_star_δ(δ) = εs(epsi-1);
                        break
                    end
                end

            end

            % In the paper it is not clearly stated how to preceed here. We
            % are looking for the smallest scale for which we can not
            % reject the null. Since we can not prefer one δ
            % neighbourhood-size, we should take the maximum of all
            % smallest scales.

%             ε_star(k) = min(ε_star_δ);
            ε_star(k) = max(ε_star_δ);
        end

        % average over all fiducial points

        % continuity statistic
        ε_min_avrg(tau_counter) = mean(ε_star);
        % undersampling statistic
        Γ(tau_counter) = mean(gamma_k);

        % increase index counter for the tau value
        tau_counter = tau_counter + 1;
    end

    %%%% for the final implementation, where we allow for a multivariate
    %%%% input dataset we here need to perform the above procedure for ALL time
    %%%% series and then pick the one, where we find 'ε_min_avrg' is
    %%%% maximal.


    % save all ε min vals corresponding to the different tau-vals for
    % this dimension-iteration
    ε_mins{cnt} = ε_min_avrg;
    % save all gamma vals corresponding to the different tau-vals for
    % this dimension-iteration
    gammas{cnt} = Γ;


    % Now we have to determine the optimal tau value from the continuitiy
    % statistic. In the paper it is not clearly stated how to achieve
    % that. They state: "If possible we choose  at a local maximum of 
    % 'ε_min_avrg' to assure the most independent coordinates
    % as in Eq. 1. If 'ε_min_avrg' remains small out to large , we
    % do not need to add more components;"

    % So we decided to look first look for the local maxima (pks are the
    % total values and locs the corresponding indices):
    [pks,locs] = findpeaks(ε_min_avrg,'MinPeakDistance',2);
    % now we pick the first local maximum, for which the preceeding and
    % succeeding peak are are smaller.
    chosen_peak = 0;
    for i = 2:length(pks)-1
        if pks(i)>pks(i-1) && pks(i)>pks(i+1)
            % we save the chosen peak with its amplitude (pks) and its
            % index (locs)
            chosen_peak = [pks(i),locs(i)];
            break
        end
    end
    % If there has not been any peak chosen in the last for loop, we simply
    % take the maximum of all values.
    if ~chosen_peak
        % look for the largest one
        [~,maxind] = max(pks);
        % save the chosen peak with its amplitude (pks) and its
        % index (locs)
        chosen_peak = [pks(maxind),locs(maxind)];
    end

    % now assign the tau value to this peak, specifically to the
    % corresponding index
    tau_use = chosen_peak(2)-1; % minus 1, because 0 is included

    % construct phase space vector according to the tau value which
    % determines the local maximum of the statistic
    V_old = embed2(V_old,s,tau_use);

    % add the chosen tau value to the output variable
    tau_vals = [tau_vals tau_use];

    % break criterions (as mentioned, there is no criterion stated in the
    % paper. They say: "If 'ε_min_avrg' remains small out to large
    % tau, we do not need to add more components; we are done and δ=d."
    % I have interpreted this in the following way: "remaining small out to
    % large tau means it has a vanishing variability, thus a small standard
    % variation (see Fig. 2 in the paper). Therefore I'll :

    % 1) break, if the standard deviation of the ε^*-curve right to
    % the chosen peak is less than break_percentage*reference_std, which is
    % the standard deviation right to the chosen peak for the very first
    % ε^*-curve. 'break_percentage' is an input parameter at the
    % moment, but if one could find a "decent" value for this I would just
    % fix this and don't leave it to the user.

%     if cnt == 1 % i.e. dimension 1
%
%         % compute std from ε-star curve right of the chosen maximum for
%         % the first curve
%         reference_curve = ε_min_avrg(chosen_peak(2):end);
%         reference_std = std(reference_curve);
%
%     else
%
%         % compute std from ε-star curve right of the chosen maximum
%         curve = ε_min_avrg(chosen_peak(2):end);
%         curve_std = std(curve);
%
%         % compare this standard deviation to the reference one
%         if curve_std < break_percentage*reference_std
%             flag = false;
%         end
%     end

    % 2) break, if the undersampling statistic cuts through the chosen β
    % level

%     if max(Γ) > β
%         flag = false;
%     end


    %%%% this is for testing the code, specifically for reproducing the
    %%%% results shown in Fig. 2 and Fig. 5. We force to break after the
    %%%% 4th embedding dimension has been reached
    if cnt == 4
        flag = false;
    end

    % increase dimension index counter
    cnt = cnt + 1;
end
% Output
V_tot = V_old;

end


%% Undersampling statistics function

function [bool,gamma] = undersampling(varargin)

% [bool,gamma] = undersampling(x1,x2,ε,β) computes the
% undersampling statistic gamma for two input time series 'x1' and 'x2' of
% the distance 'ε' under the confidence level 'β' (optional
% input, Default: β=0.05).
% It is possible to input a vector 'ε', containing a number of
% distances. In this case, the output variables are of the same length as
% the 'ε'-vector. The logical output 'bool' determines whether the
% null hypothesis gets rejected under the confidence level 'β'
% (bool = true) or not (bool = false).
%
% K.H.Kraemer, Mar 2020

%% Assign input
x1 = varargin{1};
% normalize time series
x1 = (x1-mean(x1))/std(x1);

if size(x1,1)>size(x1,2)
    x1 = x1';
end
if size(x1,1)~=1
    error('only univariate time series allowed for input')
end

x2 = varargin{2};
% normalize time series
x2 = (x2-mean(x2))/std(x2);

if size(x2,1)>size(x2,2)
    x2 = x2';
end
if size(x2,1)~=1
    error('only univariate time series allowed for input')
end

ε = varargin{3};
if size(ε,1) ~= 1 && size(ε,2) ~=1
    error('provide a valid distance vector. - This is either a column or line vector.')
end
if length(ε) == 1
    dist_vec = false;
    if ε<0
        error('provide a valid distance to test against (positive float oder int)')
    end
else
    dist_vec = true;
    for i = 1:length(ε)
        if ε(i)<0
            error('provide a valid distance vector to test against (positive float oder int)')
        end
    end
end
try
    β = varargin{4};
    if β<0 || β>1
        error('choose a valid confidence level as a float from [0 1].')
    end
catch
    β = 0.05;
end


%% estimate probability density function from input time series and conv

if range(x1)>range(x2)
    % first time series
    [hist1, edges1] = histcounts(x1,'Normalization','pdf');

    % second time series
    [hist2, ~] = histcounts(x2,edges1,'Normalization','pdf');
else
    % first time series
    [hist2, edges2] = histcounts(x2,'Normalization','pdf');

    % second time series
    [hist1, edges1] = histcounts(x1,edges2,'Normalization','pdf');
end

% construct domains
binwidth1 = mean(diff(edges1));
xx1 = (edges1(1)+binwidth1/2):binwidth1:(edges1(end)-binwidth1/2);

% convolute these distributions
σ = conv(hist1,hist2);
% normalize the distribution
σ = σ/sum(σ);

% construct domain of the convoluted signal
xx2 = ((edges1(1)+binwidth1/2)-(floor(length(xx1)/2)*binwidth1)):binwidth1:...
    ((edges1(end)-binwidth1/2)+(floor(length(xx1)/2)*binwidth1));

% truncate domain by the last point due to the convolution
if size(xx2,2) == size(σ,2) + 1
    xx2 = xx2(1:end-1);
end

% make a high resolution s-axis in the limits of the convolution support,
% in order to approximate the probabilities of finding a certain distance
xx22 = linspace(xx2(1),xx2(end),2*max(length(x1),length(x2)));

% interp the convolution signal to these high resolution points
σ2 = interp1(xx2,σ,xx22);
% normalize the distribution
σ2 = σ2/sum(σ2);


%% probability to find a distance of less than or equal to ε

if dist_vec
    gamma = zeros(1,length(ε));
    bool = zeros(1,length(ε));
    for i = 1:length(ε)
        % find indices in the convolution signal corresponding to the
        % distance input 'ε'
        n1 = find(xx22<ε(i));
        upper = n1(end);
        n2 = find(xx22>-ε(i));
        lower = n2(1);

        % compute gamma statistic
        gamma(i) = 0.5 * sum(σ2(lower:upper));

        % compare to input confidence level
        if gamma(i) < β
            bool(i) = true;
        else
            bool(i) = false;
        end
    end

else
    % find indices in the convolution signal corresponding to the distance
    % input 'ε'
    n1 = find(xx22<ε);
    upper = n1(end);
    n2 = find(xx22>-ε);
    lower = n2(1);

    % compute gamma statistic
    gamma = 0.5 * sum(σ2(lower:upper));

    % compare to input confidence level
    if gamma < β
        bool = true;
    else
        bool = false;
    end

end

end


%% Helper functions

function V2 = embed2(varargin)
% embed2 takes a matrix 'V' containing all phase space vectors, a univariate
% time series 's' and a tau value 'tau' as input. embed2 then expands the
% input phase space vectors by an additional component consisting of the
% tau-shifted values of the input time series s.
%
% V2 = embed2(V,s,tau)
%
% K.H.Kraemer, Mar 2020

V = varargin{1};
s = varargin{2};
tau = varargin{3};

if size(V,1)<size(V,2)
    V = V';
end

if size(s,1)<size(s,2)
    s = s';
end

N = size(V,1);

timespan_diff = tau;
M = N - timespan_diff;

V2 = zeros(M,size(V,2)+1);
V2(:,1:size(V,2)) = V(1:M,:);

V2(:,size(V,2)+1) = s(1+tau:N);

end

function [distances,comp_dist] = all_distances(varargin)
% all_distances2 computes all componentwise distances from one point
% (a vector) to all other given points, but not all pairwise distances
% between all points.
% This function is meant to determine the neighbourhood of a certain point
% without computing the whole distances matrix (as being done by the
% pdist()-function)
%
% [distances, comp_dist] = all_distances(fid_point,V,norm) computes all
% distances, based from the input vector 'fid_point' to all other
% points/vectors stored in the input 'V' and stores it in output
% 'distances'. The componentwise distances are stored in output 'comp_dist'
%
%
% K.H.Kraemer, Mar 2020

fid_point = varargin{1};
V = varargin{2};


methLib={'euc','max'}; % the possible norms
try
    meth = varargin{3};
    if ~isa(meth,'char') || ~ismember(meth,methLib)
       warning(['Specified norm should be one of the following possible values:',...
           10,sprintf('''%s'' ',methLib{:})])
    end
catch
    meth = 'euc';
end


% compute distances to all other points
% VV = zeros(size(V));
% for p = 1:size(VV,1)
%     VV(p,:) = fid_point;
% end
VV = repmat(fid_point,size(V,1),1);
if strcmp(meth,'euc')
    distances = sqrt(sum((VV - V) .^ 2, 2));
elseif strcmp(meth,'max')
    distances = max(abs(VV - V),[],2);
end

if nargout > 1
    comp_dist = abs(VV - V);
end

end

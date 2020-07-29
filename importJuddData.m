% Get JSON data out of the MATLAB fixation data in the Judd 2009 dataset
% modified from the original ShowEyeDataForImage function

function importJuddData(folder, numFix)

% Tilke Judd June 26, 2008
% ShowEyeDataForImage should show the eyetracking data for all users in
% 'users' on a specified image.

% modified Dec 9, 2019 by Abe Leite to export data as JSON-compatible arrays.

users = {'CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb'};

colors = cell(8, 1);
colors{1} = 'r'; colors{2} = 'g'; colors{3} = 'b'; colors{4} = 'y';
colors{5} = 'm'; colors{6} = 'c'; colors{7} = 'w'; colors{8} = 'k';
colors{9} = 'r'; colors{10} = 'g'; colors{11} = 'b'; colors{12} = 'y';
colors{13} = 'm'; colors{14} = 'c'; colors{15} = 'w'; colors{16} = 'k';

% Cycle through all images
files = dir(strcat(folder, '/*.jpeg'));
for i = 1:length(files)
    % Pick image
    filename = files(i).name;
    % Open output file
    file_id = fopen(strcat('out/', filename, '.json'), 'w');
    fprintf(file_id,'[\n');

    for j = 1:length(users)
        disp(j);
        user = users{j};

        % Get eyetracking data for this image
        datafolder = ['../DATA/' user];

        datafile = strcat(filename(1:end-4), 'mat');
        load(fullfile(datafolder, datafile));
        stimFile = eval([datafile(1:end-4)]);
        eyeData = stimFile.DATA(1).eyeData;
        [eyeData Fix Sac] = checkFixations(eyeData);

        cap = min(numFix, length(Fix.medianXY));
        numFixes = size(Fix.medianXY);
        if (numFixes(1)>=2)
            user;
            % skip the first fixation, done in original code.
            appropFix = floor(Fix.medianXY(2:cap, :));
            disp(appropFix);
            tp = transpose(appropFix);
            % there are still some negative values, to be removed later.
            fprintf(file_id, ' [%d, %d],\n', tp-1);

        end
    end
    fprintf(file_id,']\n');
    fclose(file_id);
end

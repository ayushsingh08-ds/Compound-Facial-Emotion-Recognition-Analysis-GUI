function emotionRecognitionLive()
    % Emotions list (update to match your model's order)
    emotions = { ...
        'Angrily Disgusted', ...
        'Angrily Surprised', ...
        'Disgustedly Surprised', ...
        'Happily Disgusted', ...
        'Happily Surprised', ...
        'Sadly Angry', ...
        'Sadly Fearful'};

    % Load CNN model
    modelData = load('EmotionRecognitionModel.mat');
    net = modelData.net;

    % Setup webcam
    cam = webcam();

    % Setup face detector
    faceDetector = vision.CascadeObjectDetector();

    % Create figure for GUI
    hFig = figure('Name', 'Live Emotion Recognition', 'NumberTitle', 'off', ...
                  'MenuBar', 'none', 'ToolBar', 'none', 'Color', [0.2 0.2 0.2], ...
                  'Position', [100 100 1100 600]);

    % Axes for video
    hAxVideo = axes('Parent', hFig, 'Units', 'normalized', ...
        'Position', [0.03 0.30 0.54 0.65]); % Large camera input area
    axis off

    % Axes for confidence bar
    hAxBar = axes('Parent', hFig, 'Units', 'normalized', ...
        'Position', [0.60 0.55 0.36 0.35]);
    title(hAxBar, 'Emotion Confidence', 'FontSize', 13, 'Color', [1 1 1]);
    ylim(hAxBar, [0 1]);
    hAxBar.XTick = 1:numel(emotions);
    hAxBar.XTickLabel = emotions;
    hAxBar.XTickLabelRotation = 25;
    hAxBar.FontSize = 11;
    hAxBar.XColor = [1 1 1];
    hAxBar.YColor = [1 1 1];
    hAxBar.Color = [0.16 0.16 0.16];
    ylabel(hAxBar, 'Confidence', 'Color', [1 1 1], 'FontSize', 12);

    % Emotion panel BELOW the video input
    hPanel = uipanel('Parent', hFig, 'Units', 'normalized', ...
        'Position', [0.03 0.14 0.54 0.13], ...
        'BackgroundColor', [0.1 0.1 0.1], ...
        'BorderType', 'line', ...
        'HighlightColor', [0.7 0.7 0.7]);

    hTextEmotion = uicontrol('Style', 'text', 'Parent', hPanel, ...
        'Units', 'normalized', 'Position', [0.03 0.15 0.94 0.7], ...
        'FontSize', 18, 'FontWeight', 'bold', 'ForegroundColor', 'w', ...
        'BackgroundColor', [0.1 0.1 0.1], 'String', 'Emotion: --', ...
        'HorizontalAlignment', 'center');

    % Axes for emoji image (optional, right side of emotion panel)
    hAxEmoji = axes('Parent', hPanel, 'Units', 'normalized', ...
        'Position', [0.82 0.08 0.15 0.80]);
    axis off

    % Button to capture screenshot
    btnCapture = uicontrol('Style', 'pushbutton', 'String', 'Capture Screenshot', ...
        'Parent', hFig, 'Units', 'normalized', 'Position', [0.03 0.05 0.15 0.06], ...
        'FontSize', 12, 'Callback', @captureScreenshot);

    % Variables for history
    emotionHistory = strings(1,10);
    confidenceHistory = zeros(1,10);
    historyIdx = 0;

    % Main loop control
    isRunning = true;
    set(hFig, 'CloseRequestFcn', @closeFigure);

    % Main loop
    while isRunning
        img = snapshot(cam);
        imgGray = rgb2gray(img);

        % Detect faces
        bboxes = step(faceDetector, imgGray);

        if ~isempty(bboxes)
            % Pick largest face
            [~, idx] = max(bboxes(:,3).*bboxes(:,4));
            bbox = bboxes(idx,:);

            % Crop and resize face for CNN input
            faceImg = imcrop(imgGray, bbox);
            faceResized = imresize(faceImg, [48 48]);
            faceInput = im2single(reshape(faceResized, [48 48 1]));

            % CNN prediction
            [label, scores] = classify(net, faceInput);
            cnnEmotion = emotions{double(label)};
            cnnConfidence = max(scores);

            % Facial structure inference
            structureEmotion = inferEmotionFromLandmarks(faceImg);

            % Combine CNN + structure for final decision
            finalEmotion = combineModelAndStructure(cnnEmotion, structureEmotion);

            % Update history
            historyIdx = mod(historyIdx, 10) + 1;
            emotionHistory(historyIdx) = finalEmotion;
            confidenceHistory(historyIdx) = cnnConfidence;

            % Overlay bbox and final emotion
            img = insertObjectAnnotation(img, 'rectangle', bbox, finalEmotion, ...
                                        'Color', emotionColor(finalEmotion), ...
                                        'FontSize', 14, 'TextColor', 'white');

            % Update dynamic color panel & label
            hPanel.BackgroundColor = emotionColor(finalEmotion);
            hTextEmotion.String = ['Emotion: ' finalEmotion];

            % Update confidence bar
            bar(hAxBar, scores, 'FaceColor', 'flat');
            hAxBar.XTick = 1:numel(emotions);
            hAxBar.XTickLabel = emotions;
            hAxBar.XTickLabelRotation = 25;
            hAxBar.YLim = [0 1];
            hAxBar.Color = [0.16 0.16 0.16];
            hAxBar.XColor = [1 1 1];
            hAxBar.YColor = [1 1 1];
            hAxBar.FontSize = 11;
            ylabel(hAxBar, 'Confidence', 'Color', [1 1 1], 'FontSize', 12);
            for i = 1:numel(emotions)
                hAxBar.Children.CData(i,:) = emotionColor(emotions{i});
            end

            % Optionally highlight the predicted emotion's bar:
            [~,maxIdx] = max(scores);
            hAxBar.Children.CData(maxIdx,:) = [0 0 0]; % Black highlight

            % Show emoji (optional)
            emojiFile = [finalEmotion '.png'];
            if exist(emojiFile, 'file')
                emojiImg = imread(emojiFile);
                imshow(emojiImg, 'Parent', hAxEmoji);
            else
                cla(hAxEmoji);
            end

            % Text-to-Speech (uncomment if you want voice)
            % textToSpeech(finalEmotion);
        else
            % No face found
            hTextEmotion.String = 'No face detected';
            hPanel.BackgroundColor = [0.1 0.1 0.1];
            cla(hAxEmoji);
            cla(hAxBar);
            imshow(img, 'Parent', hAxVideo);
        end

        % Show frame
        imshow(img, 'Parent', hAxVideo);

        pause(0.05); % Control frame rate
        drawnow;
    end

    % Cleanup
    clear cam;

    % --- Callback: Capture screenshot and save image with label
    function captureScreenshot(~, ~)
        frame = getframe(hAxVideo);
        imgCaptured = frame.cdata;
        filename = ['Emotion_' datestr(now, 'yyyymmdd_HHMMSS') '.png'];
        imwrite(imgCaptured, filename);
        msgbox(['Screenshot saved: ' filename]);
    end

    % --- Callback: close figure cleanly
    function closeFigure(~, ~)
        isRunning = false;
        delete(hFig);
    end

    % --- Facial landmark inference function (simple proxy)
    function emotion = inferEmotionFromLandmarks(faceImg)
        points = detectMinEigenFeatures(faceImg);
        if points.Count < 10
            emotion = 'Unknown';
            return;
        end

        locations = points.Location;
        yMean = mean(locations(:,2));
        mouthPts = locations(locations(:,2) > yMean + 5, :);
        eyePts = locations(locations(:,2) < yMean - 5, :);

        mouthOpen = ~isempty(mouthPts) && (range(mouthPts(:,2)) > 10);
        eyesWide = ~isempty(eyePts) && (range(eyePts(:,2)) > 5);

        if mouthOpen && eyesWide
            emotion = 'Happily Surprised';
        elseif ~mouthOpen && ~eyesWide
            emotion = 'Sadly Angry';
        else
            emotion = 'Disgustedly Surprised';
        end
    end

    % --- Combine CNN and structure emotions simply
    function final = combineModelAndStructure(modelPred, structPred)
        if strcmp(structPred, 'Unknown')
            final = modelPred;
        elseif contains(lower(modelPred), lower(structPred))
            final = modelPred;
        else
            final = [modelPred ' + ' structPred];
        end
    end

    % --- Map emotions to color for panel & annotation
    function c = emotionColor(em)
        switch em
            case {'Angrily Disgusted', 'Sadly Angry'}
                c = [1 0 0]; % Red
            case {'Angrily Surprised', 'Disgustedly Surprised'}
                c = [1 0.5 0]; % Orange
            case {'Happily Disgusted', 'Happily Surprised'}
                c = [0 1 0]; % Green
            case {'Sadly Fearful'}
                c = [0 0 1]; % Blue
            otherwise
                c = [0.5 0.5 0.5]; % Gray
        end
    end

    % --- Optional: text to speech (Windows PowerShell example)
    function textToSpeech(txt)
        % Requires Windows PowerShell available
        command = ['PowerShell -Command "Add-Type -AssemblyName System.Speech; ', ...
            '$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;', ...
            '$speak.Speak(''' txt ''')"'];
        system(command);
    end
end
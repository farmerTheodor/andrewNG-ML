function transformedY = transformY(y, numLabels)
  transformedY = [];
  for i = 1:rows(y)
    current = zeros(1,numLabels);
    current(y(i)) = 1;
    transformedY = [transformedY;current];
  endfor
  transformedY;
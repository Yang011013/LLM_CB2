UNITY="C:/Program Files/Unity/Editor/Unity"
$UNITY -quit -batchmode -logFile - -projectPath game/ -executeMethod WebGLBuilder.Build
rm -rf server/www/OLD_WebGL
mv server/www/WebGL server/www/OLD_WebGL
cp -rf game/builds/WebGLVersion server/www/WebGL
echo "Moved build to server/www directory"

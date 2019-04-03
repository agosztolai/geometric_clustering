function v = createMovie( getframe, v, closefile )
% This script creates a movie of a sequence of snapshots
% getframe must be on object obtained with getframe(gcf)
% closefile == 1 will terminate movie

if v == 0
    v = VideoWriter([pwd,'/video']);   
    v.FrameRate = 5; 
    open(v)
end

writeVideo(v,getframe);

if closefile == 1
    close(v)
end

end
% 2023B第二问 三维几何示意图 (对应题目图8)
% 要素: 水平面/坡面/测线及其投影/法向及其投影/夹角β
%       测量截面/波束边缘/坡面覆盖带/覆盖宽度W
% 输出: ../fig/fig_geometry3d.png
% 用法: matlab -batch "fig_geometry3d"
function fig_geometry3d
    out = fullfile(fileparts(mfilename('fullpath')), '..', 'fig', 'fig_geometry3d.png');

    ALPHA = deg2rad(15);   % 示意图坡度(夸张, 真实α=1.5°)
    BETA = deg2rad(45);    % 示意夹角β

    fig = figure('Position', [100 100 950 720], 'Color', 'w');
    ax = axes('Parent', fig);

    % ===== 水平面 =====
    [xx, yy] = meshgrid(linspace(-2.4, 2.4, 12));
    surf(ax, xx, yy, zeros(size(xx)), 'FaceColor', [0.18 0.53 0.67], ...
        'FaceAlpha', 0.12, 'EdgeColor', 'none');
    hold(ax, 'on');

    % ===== 海底坡面 z = y*tan(ALPHA) =====
    [xx, yy] = meshgrid(linspace(-2.4, 2.4, 25));
    surf(ax, xx, yy, yy * tan(ALPHA), 'FaceColor', [0.55 0.27 0.07], ...
        'FaceAlpha', 0.5, 'EdgeColor', 'none');

    % ===== 测线方向t与法向 =====
    t = [sin(BETA), -cos(BETA), 0];            % 测线方向(水平)
    n = [-t(2), t(1), 0];                      % 垂直测线方向(水平)
    N = [0, -tan(ALPHA), 1]; N = N / norm(N);  % 坡面法向(向上)

    % ===== 船与测量截面 =====
    p0 = [0, 0, 1.6];                          % 船位
    p0b = [0, 0, 0];                           % 船正下方坡面点
    % 测量截面(垂直测线, 过船): 矩形 [p0, p0±1.5n, 向下1.2]
    sec = [p0 + 1.5*n; p0 - 1.5*n; p0 - 1.5*n - [0 0 1.3]; p0 + 1.5*n - [0 0 1.3]];
    patch(ax, sec(:,1), sec(:,2), sec(:,3), [0.9 0.95 1], ...
        'FaceAlpha', 0.35, 'EdgeColor', [0.3 0.4 0.6], 'LineWidth', 0.8);

    % ===== 波束边缘(截面内与竖直60°)与坡面交点 =====
    for sgn = [1 -1]
        dir = [sgn*0.866*n(1), sgn*0.866*n(2), -0.5];  % 与竖直成60°
        s = (p0b(2)*tan(ALPHA) - p0(3)) / (dir(3) - dir(2)*tan(ALPHA));
        P = p0 + s * dir;
        if sgn == 1, PL = P; else, PR = P; end
        plot3(ax, [p0(1) P(1)], [p0(2) P(2)], [p0(3) P(3)], ...
            'b-', 'LineWidth', 2);
    end

    % ===== 坡面覆盖带(两落点间坡面条带) =====
    xb = linspace(PL(1), PR(1), 20);
    yb = linspace(PL(2), PR(2), 20);
    patch(ax, [xb fliplr(xb)], [yb fliplr(yb)], ...
        [yb*tan(ALPHA) fliplr((yb+0.6)*tan(ALPHA))], [0.1 0.6 0.9], ...
        'FaceAlpha', 0.45, 'EdgeColor', 'none');
    % 覆盖带纵向延伸一段(沿测线方向)
    for dlt = [0 0.5]
        patch(ax, [PL(1)+dlt*t(1) PR(1)+dlt*t(1) PR(1)+dlt*t(1) PL(1)+dlt*t(1)], ...
              [PL(2)+dlt*t(2) PR(2)+dlt*t(2) PR(2)+dlt*t(2) PL(2)+dlt*t(2)], ...
              [(PL(2)+dlt*t(2))*tan(ALPHA) (PR(2)+dlt*t(2))*tan(ALPHA) ...
               (PR(2)+dlt*t(2))*tan(ALPHA) (PL(2)+dlt*t(2))*tan(ALPHA)], ...
              [0.1 0.6 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    end
    % 落点连线(坡面上)
    plot3(ax, [PL(1) PR(1)], [PL(2) PR(2)], [PL(2)*tan(ALPHA) PR(2)*tan(ALPHA)], ...
        'b-', 'LineWidth', 2.5);

    % ===== W标注 =====
    W_h = norm([PR(1)-PL(1), PR(2)-PL(2)]);
    mid = (PL + PR) / 2;
    plot3(ax, [PL(1) PL(1)],[PL(2) PL(2)],[PL(2)*tan(ALPHA) PL(2)*tan(ALPHA)+0.45], ...
        'k-', 'LineWidth', 1);
    plot3(ax, [PR(1) PR(1)],[PR(2) PR(2)],[PR(2)*tan(ALPHA) PR(2)*tan(ALPHA)+0.45], ...
        'k-', 'LineWidth', 1);
    plot3(ax, [PL(1) PR(1)], [PL(2) PR(2)], ...
        [PL(2)*tan(ALPHA)+0.45 PR(2)*tan(ALPHA)+0.45], 'k-', 'LineWidth', 1.5);
    text(ax, mid(1), mid(2), mid(2)*tan(ALPHA)+0.6, sprintf('W = %.1f', W_h), ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');

    % ===== 测线及其投影 =====
    p1 = p0 + t * 3.0;
    plot3(ax, [p0(1) p1(1)], [p0(2) p1(2)], [p0(3) p1(3)], 'k-', 'LineWidth', 2.5);
    quiver3(ax, p1(1), p1(2), p1(3), t(1)*0.4, t(2)*0.4, 0, 0, ...
        'k', 'LineWidth', 2.5, 'MaxHeadSize', 0.8);
    plot3(ax, [p0(1) p1(1)], [p0(2) p1(2)], [0 0], 'k--', 'LineWidth', 1.5);
    plot3(ax, [p0(1) p0(1)], [p0(2) p0(2)], [0 1.6], 'k:', 'LineWidth', 1);
    plot3(ax, p0(1), p0(2), p0(3), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    text(ax, p0(1)+0.15, p0(2)+0.15, p0(3)+0.2, '船', 'FontSize', 12);

    % ===== 坡面法向及其投影 =====
    pt = [0.6, 0.9, 0.9*tan(ALPHA)];
    q = pt + N * 1.6;
    plot3(ax, [pt(1) q(1)], [pt(2) q(2)], [pt(3) q(3)], 'r-', 'LineWidth', 2.5);
    quiver3(ax, q(1), q(2), q(3), N(1)*0.35, N(2)*0.35, N(3)*0.35, 0, ...
        'r', 'LineWidth', 2.5, 'MaxHeadSize', 0.8);
    plot3(ax, [pt(1) pt(1)], [pt(2) pt(2)], [0 pt(3)], 'g--', 'LineWidth', 1.2);
    plot3(ax, [pt(1) pt(1)+N(1)*1.6], [pt(2) pt(2)+N(2)*1.6], [0 0], ...
        'r--', 'LineWidth', 1.8);
    text(ax, pt(1)-0.1, pt(2)-0.4, pt(3)+0.15, '坡面法向', 'Color', 'r', 'FontSize', 11);
    text(ax, pt(1)+0.25, pt(2)-0.65, 0.05, '法向投影', 'Color', 'r', 'FontSize', 11);

    % ===== β角弧 =====
    a1 = atan2(t(2), t(1));
    a2 = -pi/2;
    th = linspace(min(a1,a2), max(a1,a2), 40);
    r = 0.9;
    plot3(ax, r*cos(th), r*sin(th), zeros(size(th)), 'm-', 'LineWidth', 2);
    text(ax, r*1.3*cos((a1+a2)/2), r*1.3*sin((a1+a2)/2), 0.12, '\beta', ...
        'Color', 'm', 'FontSize', 16, 'FontWeight', 'bold');

    % ===== α角标注 =====
    plot3(ax, [0.4 1.1], [0.4 0.4], [0 0], '-', 'Color', [0.44 0.19 0.63], 'LineWidth', 2);
    plot3(ax, [1.1 1.1], [0.4 0.4], [0 0.7*tan(ALPHA)], '-', ...
        'Color', [0.44 0.19 0.63], 'LineWidth', 2);
    plot3(ax, [0.4 1.1], [0.4 0.4], [0 0.7*tan(ALPHA)], '-', ...
        'Color', [0.44 0.19 0.63], 'LineWidth', 2);
    text(ax, 1.3, 0.45, 0.3*tan(ALPHA), '\alpha', 'Color', [0.44 0.19 0.63], ...
        'FontSize', 14, 'FontWeight', 'bold');

    text(ax, -2.2, -2.5, 0, '水平面', 'Color', [0.18 0.53 0.67], 'FontSize', 12);
    text(ax, 1.6, 2.1, 2.2*tan(ALPHA)+0.1, '海底坡面', 'Color', [0.55 0.27 0.07], 'FontSize', 12);
    text(ax, 1.3, -0.2, 1.95, '测线方向', 'FontSize', 12);

    xlim(ax, [-2.4 2.4]); ylim(ax, [-2.4 2.4]); zlim(ax, [0 2.2]);
    xlabel(ax, 'x'); ylabel(ax, 'y (坡下为正)'); zlabel(ax, 'z');
    view(ax, -60, 22);
    grid(ax, 'on'); axis(ax, 'tight');
    hold(ax, 'off');

    exportgraphics(fig, out, 'Resolution', 300);
    fprintf('[OK] %s\n', out);
end

% 2023B第一问 截面几何示意图 (2D) — 修订版
% 修正: 仅保留60°半开角/α弧在坡面与水平面交角处/垂距D标注/
%       几何元素对齐原点/左右深度标注/字体加大
% 输出: ../fig/fig_geometry.png
% 用法: matlab -batch "fig_geometry"
function fig_geometry
    out = fullfile(fileparts(mfilename('fullpath')), '..', 'fig', 'fig_geometry.png');

    D = 70.0; ALPHA = deg2rad(1.5); t30 = tan(pi/6);

    fig = figure('Position', [100 100 1050 520], 'Color', 'w');
    ax = axes('Parent', fig); hold(ax, 'on');
    axis(ax, 'equal');

    % ===== 海平面 =====
    plot(ax, [-165 165], [0 0], '-', 'Color', [0.12 0.31 0.47], 'LineWidth', 2.2);
    text(ax, 163, 8, '海平面', 'HorizontalAlignment', 'right', ...
        'Color', [0.12 0.31 0.47], 'FontSize', 14);

    % ===== 海底坡面 (y>0为坡下, 右深左浅) =====
    y = linspace(-160, 160, 200);
    yp = -(D + y * tan(ALPHA));
    plot(ax, y, yp, '-', 'Color', [0.55 0.27 0.07], 'LineWidth', 2.5);
    patch(ax, [y fliplr(y)], [yp fliplr(-130*ones(size(y)))], ...
        [0.82 0.71 0.55], 'FaceAlpha', 0.3, 'EdgeColor', 'none');

    % ===== 船 (原点) 与垂线 =====
    plot(ax, 0, 0, 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
    text(ax, 12, 6, '船', 'FontSize', 14, 'FontWeight', 'bold');
    plot(ax, [0 0], [0 -D], 'k--', 'LineWidth', 1.2);

    % ===== 波束边缘 (与竖直法线成60°, 即半开角) =====
    for s = [-1 1]
        yy = linspace(0, s*130, 100);
        plot(ax, yy, -abs(yy)*t30, '-', 'Color', [0.18 0.46 0.71], ...
            'LineWidth', 1.8);
    end

    % ===== 交点 P_L / P_R =====
    yL = -D/(t30 + tan(ALPHA)); yR = D/(t30 - tan(ALPHA));
    zL = D + yL*tan(ALPHA); zR = D + yR*tan(ALPHA);
    plot(ax, [yL yR], [-zL -zR], 'ro', 'MarkerSize', 7, 'MarkerFaceColor', 'r');
    text(ax, yL-45, -zL-18, 'P_L', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 14);
    text(ax, yR+6, -zR-18, 'P_R', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 14);

    % ===== 覆盖宽度 W (海平面基准) =====
    plot(ax, [yL yR], [12 12], 'r-', 'LineWidth', 1.8);
    plot(ax, [yL yL], [8 16], 'r-', 'LineWidth', 1.8);
    plot(ax, [yR yR], [8 16], 'r-', 'LineWidth', 1.8);
    text(ax, (yL+yR)/2, 28, sprintf('W = %.1f m', yR-yL), ...
        'HorizontalAlignment', 'center', 'Color', 'r', ...
        'FontSize', 15, 'FontWeight', 'bold');

    % ===== 唯一角度: 60° (半开角, 波束边缘与竖直法线夹角, 以船为圆心) =====
    th = linspace(deg2rad(-90), deg2rad(-30), 40);
    r = 20;
    plot(ax, r*cos(th), r*sin(th), '-', 'Color', [0.18 0.46 0.71], ...
        'LineWidth', 1.5);
    text(ax, r*1.35*cos(deg2rad(-60)), r*1.35*sin(deg2rad(-60)), '60°', ...
        'HorizontalAlignment', 'center', 'Color', [0.18 0.46 0.71], ...
        'FontSize', 14);

    % ===== α角: 坡面与水平面交角 (圆心=船正下方海底点) =====
    th = linspace(pi, pi - deg2rad(1.5), 30);
    r = 14;
    plot(ax, r*cos(th), -D + r*sin(th), '-', 'Color', [0.44 0.19 0.63], ...
        'LineWidth', 1.5);
    text(ax, r*1.4*cos(pi - deg2rad(0.75)), -D + r*1.4*sin(pi - deg2rad(0.75)), ...
        '\alpha=1.5°', 'Color', [0.44 0.19 0.63], 'FontSize', 14);

    % ===== 垂距 D 标注 (船正下方, 虚线法线处) =====
    plot(ax, [10 10], [0 -D], 'k-', 'LineWidth', 1.2);
    plot(ax, [7 13], [0 0], 'k-', 'LineWidth', 1.2);
    plot(ax, [7 13], [-D -D], 'k-', 'LineWidth', 1.2);
    text(ax, 16, -D/2, '垂距 D=70m', 'FontSize', 13, ...
        'VerticalAlignment', 'middle');

    % ===== 左右端深度标注 (增强坡度视觉: 右深左浅) =====
    text(ax, -160, -(D-160*tan(ALPHA)) - 10, '浅 ~65.8m', ...
        'FontSize', 12, 'Color', [0.3 0.3 0.3], 'HorizontalAlignment', 'left');
    text(ax, 160, -(D+160*tan(ALPHA)) - 10, '深 ~74.2m', ...
        'FontSize', 12, 'Color', [0.3 0.3 0.3], 'HorizontalAlignment', 'right');

    xlim([-165 165]); ylim([-110 60]);
    xlabel('y (m)  向坡下为正', 'FontSize', 13);
    ylabel('z (m)  深度(向下)', 'FontSize', 13);
    grid(ax, 'on');
    hold(ax, 'off');

    exportgraphics(fig, out, 'Resolution', 300);
    fprintf('[OK] %s\n', out);
end

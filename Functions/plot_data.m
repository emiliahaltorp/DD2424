
% Function to plat data
function plot_data(x, y1, y2, ylab, xlab, titletext)
    figure
    plot(x, y1,'LineWidth',3);
    hold on
    plot(x, y2,'LineWidth',3);
    xlabel(xlab,'FontSize',20)
    ylabel(ylab,'FontSize',20)
    legend('Training', 'Validation','FontSize',15)
    set(gca,'FontSize',20);
    %ylim([0 4])
    grid on
    title(titletext)
end

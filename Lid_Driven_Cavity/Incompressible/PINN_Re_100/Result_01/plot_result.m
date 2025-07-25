
%% PLotting contours
x = importdata('X.txt');
y = importdata('Y.txt');
u = importdata('u.txt');
v = importdata('v.txt');
p = importdata('p.txt');

% u velocity
figure('name','1', 'color', [1,1,1], ...
       'position',[200,200,500,400]);
contourf(x,y,u,20,'EdgeAlpha',0)
set(gca, 'fontsize',10, 'fontweight','bold');
xlabel('x/L_{ref}'); ylabel('y/L_{ref}');
colorbar; colormap("jet")
axis('equal')

% v velocity
figure('name','2', 'color', [1,1,1], ...
       'position',[200,200,500,400]);
contourf(x,y,v,20,'EdgeAlpha',0)
set(gca, 'fontsize',10, 'fontweight','bold');
xlabel('x/L_{ref}'); ylabel('y/L_{ref}');
colorbar; colormap("jet")
axis('equal')

% pressure
figure('name','3', 'color', [1,1,1], ...
       'position',[200,200,500,400]);
contourf(x,y,p,20,'EdgeAlpha',0)
set(gca, 'fontsize',10, 'fontweight','bold');
xlabel('x/L_{ref}'); ylabel('y/L_{ref}');
colorbar; colormap("jet")
axis('equal')


h = importdata('loss_history.out');
figure('name','4', 'color', [1,1,1], ...
       'position',[200,200,500,400]);
plot(h,'-k','linewidth',3)
set(gca, 'fontsize',10, 'fontweight','bold');
xlabel('Iteration'); ylabel('Loss');

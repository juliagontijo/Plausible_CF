# for Plotting
# def plot_decision_boundary(X, y, func, contrafactual_set):
#     h = 0.1
#     xmin, ymin = np.min(X, axis=0)
#     xmax, ymax = np.max(X, axis=0)

#     xx, yy = np.meshgrid(
#         np.arange(xmin, xmax, h),
#         np.arange(ymin, ymax, h)
#         )

#     cm = plt.cm.RdBu
#     cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
#     newx = np.c_[xx.ravel(), yy.ravel()]
        
#     fig, ax = plt.subplots()
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
    
#     # Z = clf.predict_proba(newx)[:, 1]
#     Z = func(newx)
#     Z = Z.reshape(xx.shape)

#     v=contour_plot = ax.contourf(
#         xx, yy,
#         Z, 
#         levels=20,
#         cmap=cm, 
#         alpha=.8)
    
#     ax.scatter(X[:, 0], X[:, 1], 
#                 c=y, 
#                 cmap=cm_bright,
#                 edgecolors='k',
#                 zorder=1)    

#      # Plot the subject (first position in X) as a green dot
#     ax.scatter(X[0, 0], X[0, 1], color='green', edgecolors='k', zorder=2, label='Subject')

#     # Plot counterfactual set individuals as black dots
#     ax.scatter(contrafactual_set.iloc[:, 0], contrafactual_set.iloc[:, 1], color='black', marker='o', label='Counterfactuals')


#     ax.grid(color='k', 
#             linestyle='-', 
#             linewidth=0.50, 
#             alpha=0.75)

#     plt.colorbar(v, ax=ax)

#     # Adding a legend to identify the subject and counterfactuals
#     ax.legend()

#     plt.show()

# def plot_decision_boundary_initial(X, y, func):
#     h = 0.1
#     xmin, ymin = np.min(X, axis=0)
#     xmax, ymax = np.max(X, axis=0)

#     xx, yy = np.meshgrid(
#         np.arange(xmin, xmax, h),
#         np.arange(ymin, ymax, h)
#         )

#     cm = plt.cm.RdBu
#     cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
#     newx = np.c_[xx.ravel(), yy.ravel()]
        
#     fig, ax = plt.subplots()
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
    
#     # Z = clf.predict_proba(newx)[:, 1]
#     Z = func(newx)
#     Z = Z.reshape(xx.shape)

#     v=contour_plot = ax.contourf(
#         xx, yy,
#         Z, 
#         levels=20,
#         cmap=cm, 
#         alpha=.8)
    
#     ax.scatter(X[:, 0], X[:, 1], 
#                 c=y, 
#                 cmap=cm_bright,
#                 edgecolors='k',
#                 zorder=1)    

#      # Plot the subject (first position in X) as a green dot
#     ax.scatter(X[0, 0], X[0, 1], color='green', edgecolors='k', zorder=2, label='Subject')

#     ax.grid(color='k', 
#             linestyle='-', 
#             linewidth=0.50, 
#             alpha=0.75)

#     plt.colorbar(v, ax=ax)

#     # Adding a legend to identify the subject and counterfactuals
#     ax.legend()
#     ax.set_title("Initial Population")

#     plt.show()

# def plot_TSNE(X, y, perpl):

#     scaler = preprocessing.MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     t_sne = manifold.TSNE(
#         init = 'pca',
#         n_components=2, #make it not hard coded
#         perplexity=perpl,
#         n_iter=2000,
#         #early_exag_coeff = 12,
#         #stop_lying_iter = 100
#         random_state=0
#     )

#     S_t_sne = t_sne.fit_transform(X_scaled)
#     print(t_sne.kl_divergence_)


#     # Plot 1
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(S_t_sne[:, 0], S_t_sne[:, 1], c=y, cmap='viridis', alpha=0.5)
#     plt.colorbar(scatter)
#     plt.title('t-SNE visualization of original and contrafactual data')
#     plt.show()


#     # # Plot 2
#     # fig = px.scatter(x=S_t_sne[:, 0], y=S_t_sne[:, 1], color=y)
#     # fig = px.scatter(x=S_t_sne[:, 0], y=S_t_sne[:, 1], color=y)
#     # fig.update_layout(
#     #     title="t-SNE visualization of Custom Classification dataset",
#     #     xaxis_title="First t-SNE",
#     #     yaxis_title="Second t-SNE",
#     # )
#     # fig.show()


#     # # Plot 3
#     # plot_2d(S_t_sne, y, "T-distributed Stochastic  \n Neighbor Embedding")

# def plot_ISOMAP(X, y):
#     scaler = preprocessing.MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)

#     m_isomap = manifold.Isomap(
#         n_components=2 #make it not hard coded
#     )

#     S_t_sne = m_isomap.fit_transform(X_scaled)
#     print(m_isomap.kernel_pca_)

#     # Plot 1
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(S_t_sne[:, 0], S_t_sne[:, 1], c=y, cmap='viridis', alpha=0.5)
#     plt.colorbar(scatter)
#     plt.title('Isomap visualization of original and contrafactual data')
#     plt.show()


#     # # Plot 2
#     # fig = px.scatter(x=S_t_sne[:, 0], y=S_t_sne[:, 1], color=y)
#     # fig = px.scatter(x=S_t_sne[:, 0], y=S_t_sne[:, 1], color=y)
#     # fig.update_layout(
#     #     title="t-SNE visualization of Custom Classification dataset",
#     #     xaxis_title="First t-SNE",
#     #     yaxis_title="Second t-SNE",
#     # )
#     # fig.show()


#     # # Plot 3
#     # plot_2d(S_t_sne, y, "T-distributed Stochastic  \n Neighbor Embedding")

# def plot_LL(X, y):
#     scaler = preprocessing.MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)

#     m_ll = manifold.LocallyLinearEmbedding(
#         n_components=2  #make it not hard coded
#     )

#     S_t_sne = m_ll.fit_transform(X_scaled)

#     # Plot 1
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(S_t_sne[:, 0], S_t_sne[:, 1], c=y, cmap='viridis', alpha=0.5)
#     plt.colorbar(scatter)
#     plt.title('LocallyLinearEmbedding visualization of original and contrafactual data')
#     plt.show()

# def plot_Spectral(X, y):
#     scaler = preprocessing.MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)

#     m_ll = manifold.SpectralEmbedding(
#         n_components=2  #make it not hard coded
#     )

#     S_t_sne = m_ll.fit_transform(X_scaled)

#     # Plot 1
#     plt.figure(figsize=(10, 6))
#     scatter = plt.scatter(S_t_sne[:, 0], S_t_sne[:, 1], c=y, cmap='viridis', alpha=0.5)
#     plt.colorbar(scatter)
#     plt.title('SpectralEmbedding visualization of original and contrafactual data')
#     plt.show()


# def plot_FreeViz(X, y, hasContrafactual, k):
#     X_scaled = normalize_if_needed(X)

#     freeviz = FreeViz()
#     projection = freeviz(X_scaled)
    
#     # Extract coordinates and class labels
#     coords = projection[:, :2]
#     classes = y[:, -1].metas[:, 0]

#     # Plot the projection
#     plt.figure(figsize=(10, 7))
#     # Plot the first instance in green
#     plt.scatter(coords[0, 0], coords[0, 1], color='green', label='First instance')

#     # Plot the rest of the instances, changing the last k instances to red if hasContrafactual is True
#     for i, cls in enumerate(set(classes)):
#         if hasContrafactual:
#             plt.scatter(coords[classes == cls, 0][1:-k], coords[classes == cls, 1][1:-k], label=cls)
#             plt.scatter(coords[classes == cls, 0][-k:], coords[classes == cls, 1][-k:], color='red', label=f'Last {k} instances')
#         else:
#             plt.scatter(coords[classes == cls, 0][1:], coords[classes == cls, 1][1:], label=cls)

#     plt.legend()
#     plt.title("FreeViz Projection of Iris Dataset")
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.show()


# # for Plotting path
# def plot_decision_boundary(X, y, func):
#     h = 0.1
#     xmin, ymin = np.min(X, axis=0)
#     xmax, ymax = np.max(X, axis=0)

#     xx, yy = np.meshgrid(
#         np.arange(xmin, xmax, h),
#         np.arange(ymin, ymax, h)
#         )

#     cm = plt.cm.RdBu
#     cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
    
#     newx = np.c_[xx.ravel(), yy.ravel()]
        
#     fig, ax = plt.subplots()
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
    
#     # Z = clf.predict_proba(newx)[:, 1]
#     Z = func(newx)
#     Z = Z.reshape(xx.shape)

#     v=contour_plot = ax.contourf(
#         xx, yy,
#         Z, 
#         levels=20,
#         cmap=cm, 
#         alpha=.8)
    
#     ax.scatter(X[:, 0], X[:, 1], 
#                 c=y, 
#                 cmap=cm_bright,
#                 edgecolors='k',
#                 zorder=1)

#     ax.grid(color='k', 
#             linestyle='-', 
#             linewidth=0.50, 
#             alpha=0.75)

#     plt.colorbar(v, ax=ax)
#     return ax


# def plot_2d(points, points_color, title):
#     fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
#     fig.suptitle(title, size=16)
#     add_2d_scatter(ax, points, points_color)
#     plt.show()

# def add_2d_scatter(ax, points, points_color, title=None):
#     x, y = points.T
#     ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
#     ax.set_title(title)
#     ax.xaxis.set_major_formatter(ticker.NullFormatter())
#     ax.yaxis.set_major_formatter(ticker.NullFormatter())


# # for Density weight
# def sigmoid(x):
#     return 1/(1 + np.exp(-x))



# def plot(self, x_train, y_train, csse_counterfactuals, original_ind):
    #     X = pd.concat([x_train, csse_counterfactuals])

    #     h = 0.1
    #     xmin, ymin = np.min(X.values, axis=0)
    #     xmax, ymax = np.max(X.values, axis=0)

    #     xx, yy = np.meshgrid(
    #         np.arange(xmin, xmax, h),
    #         np.arange(ymin, ymax, h)
    #         )

    #     cm = plt.cm.RdBu
    #     cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])    
        
    #     newx = np.c_[xx.ravel(), yy.ravel()]
            
    #     fig, ax = plt.subplots()
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
        
    #     # Z = clf.predict_proba(newx)[:, 1]
    #     Z = model.predict_proba(newx)
    #     Z = Z.reshape(xx.shape)

    #     v=contour_plot = ax.contourf(
    #         xx, yy,
    #         Z, 
    #         levels=20,
    #         cmap=cm, 
    #         alpha=.8)
        
    #     ax.scatter(X.values[:, 0], X.values[:, 1], 
    #                 c=y, 
    #                 cmap=cm_bright,
    #                 edgecolors='k',
    #                 zorder=1)

    #     ax.scatter(csse_counterfactuals.values[:, 0], csse_counterfactuals.values[:, 1], 
    #                 c='green', 
    #                 cmap=cm_bright,
    #                 edgecolors='k',
    #                 zorder=1)
        
    #     ax.scatter(original_instance.values[0, 0], original_instance.values[0, 1],
    #        c='k',
    #        edgecolors='k',
    #        zorder=1)


    #     ax.grid(color='k', 
    #             linestyle='-', 
    #             linewidth=0.50, 
    #             alpha=0.75)

    #     plt.colorbar(v, ax=ax)
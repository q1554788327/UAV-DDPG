import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def plot_trajectory_static():
    """Create static 3D trajectory plot and save"""
    try:
        # Check if files exist
        import os
        if not os.path.exists('best_trajectory.npy'):
            print("best_trajectory.npy not found! Creating sample data...")
            create_sample_data()
        
        # Load data
        trajectory = np.load('best_trajectory.npy')
        users = np.load('best_users.npy')
        
        print(f"Loaded trajectory with {len(trajectory)} points")
        print(f"Loaded {len(users)} users")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data...")
        trajectory, users = create_sample_data()
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Create 2x2 subplot layout
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d_trajectory(ax1, trajectory, users)
    
    # XY plane projection
    ax2 = fig.add_subplot(222)
    plot_xy_projection(ax2, trajectory, users)
    
    # XZ plane projection
    ax3 = fig.add_subplot(223)
    plot_xz_projection(ax3, trajectory, users)
    
    # Trajectory information table
    ax4 = fig.add_subplot(224)
    plot_trajectory_info(ax4, trajectory, users)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig('uav_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    
    # Generate detailed coordinate files
    save_coordinate_details(trajectory, users)

def plot_3d_trajectory(ax, trajectory, users):
    """Plot 3D trajectory"""
    # Plot trajectory line
    ax.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2], 
            'o-', color='orange', linewidth=2, markersize=4, alpha=0.8, label='UAV Path')
    
    # Mark key points
    ax.scatter(trajectory[0,0], trajectory[0,1], trajectory[0,2], 
               c='green', marker='o', s=100, label='Start', alpha=1.0)
    ax.scatter(trajectory[-1,0], trajectory[-1,1], trajectory[-1,2], 
               c='red', marker='x', s=150, label='End', alpha=1.0)
    
    # Plot user positions
    ax.scatter(users[:,0], users[:,1], users[:,2], 
               c='blue', marker='^', s=100, label='Users', alpha=0.8)
    
    # Add user labels
    for i, user in enumerate(users):
        ax.text(user[0], user[1], user[2]+2, f'U{i}', fontsize=8)
    
    # Add some trajectory point labels
    step_indices = np.linspace(0, len(trajectory)-1, 10, dtype=int)
    for i in step_indices:
        ax.text(trajectory[i,0], trajectory[i,1], trajectory[i,2]+1, 
                f'{i}', fontsize=6, alpha=0.7)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D UAV Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_xy_projection(ax, trajectory, users):
    """Plot XY plane projection"""
    ax.plot(trajectory[:,0], trajectory[:,1], 'o-', 
            color='orange', linewidth=2, markersize=3, alpha=0.8, label='UAV Path')
    
    # Start and end points
    ax.scatter(trajectory[0,0], trajectory[0,1], c='green', marker='o', s=100, label='Start')
    ax.scatter(trajectory[-1,0], trajectory[-1,1], c='red', marker='x', s=100, label='End')
    
    # User positions
    ax.scatter(users[:,0], users[:,1], c='blue', marker='^', s=100, label='Users')
    
    # Add user labels and coordinates
    for i, user in enumerate(users):
        ax.annotate(f'U{i}\n({user[0]:.1f},{user[1]:.1f})', 
                   (user[0], user[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('XY Plane Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

def plot_xz_projection(ax, trajectory, users):
    """Plot XZ plane projection (height profile)"""
    ax.plot(trajectory[:,0], trajectory[:,2], 'o-', 
            color='orange', linewidth=2, markersize=3, alpha=0.8, label='UAV Path')
    
    ax.scatter(trajectory[0,0], trajectory[0,2], c='green', marker='o', s=100, label='Start')
    ax.scatter(trajectory[-1,0], trajectory[-1,2], c='red', marker='x', s=100, label='End')
    ax.scatter(users[:,0], users[:,2], c='blue', marker='^', s=100, label='Users')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('XZ Plane Projection (Height Profile)')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_trajectory_info(ax, trajectory, users):
    """Display trajectory information table"""
    ax.axis('off')
    
    # Calculate statistics
    total_distance = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
    max_height = np.max(trajectory[:,2])
    min_height = np.min(trajectory[:,2])
    
    info_text = f"""
Trajectory Analysis:

Total Steps: {len(trajectory)}
Total Flight Distance: {total_distance:.2f} m
Max Height: {max_height:.2f} m  
Min Height: {min_height:.2f} m

Start Point: ({trajectory[0,0]:.2f}, {trajectory[0,1]:.2f}, {trajectory[0,2]:.2f})
End Point: ({trajectory[-1,0]:.2f}, {trajectory[-1,1]:.2f}, {trajectory[-1,2]:.2f})

User Positions:
"""
    
    for i, user in enumerate(users):
        info_text += f"User{i}: ({user[0]:.2f}, {user[1]:.2f}, {user[2]:.2f})\n"
    
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def save_coordinate_details(trajectory, users):
    """Save detailed coordinate information to files"""
    # Save trajectory coordinates
    with open('trajectory_coordinates.txt', 'w') as f:
        f.write("UAV Trajectory Coordinates\n")
        f.write("Step\tX(m)\tY(m)\tZ(m)\tDistance_from_prev\n")
        f.write("-" * 50 + "\n")
        
        prev_pos = None
        for i, pos in enumerate(trajectory):
            if prev_pos is not None:
                dist = np.linalg.norm(pos - prev_pos)
            else:
                dist = 0.0
            
            f.write(f"{i}\t{pos[0]:.3f}\t{pos[1]:.3f}\t{pos[2]:.3f}\t{dist:.3f}\n")
            prev_pos = pos
    
    # Save user coordinates
    with open('user_coordinates.txt', 'w') as f:
        f.write("User Coordinates\n")
        f.write("User_ID\tX(m)\tY(m)\tZ(m)\n")
        f.write("-" * 30 + "\n")
        
        for i, user in enumerate(users):
            f.write(f"{i}\t{user[0]:.3f}\t{user[1]:.3f}\t{user[2]:.3f}\n")
    
    print("Saved detailed coordinates to 'trajectory_coordinates.txt' and 'user_coordinates.txt'")

def create_sample_data():
    """Create sample data"""
    print("Creating sample trajectory data...")
    
    # Sample trajectory (spiral path)
    t = np.linspace(0, 4*np.pi, 100)
    x = 10 * np.cos(t) + 25
    y = 10 * np.sin(t) + 25  
    z = np.linspace(10, 50, 100)
    
    trajectory = np.column_stack([x, y, z])
    
    # Sample user positions
    users = np.array([
        [15, 15, 0],
        [35, 35, 0], 
        [25, 40, 0]
    ])
    
    # Save sample data
    np.save('best_trajectory.npy', trajectory)
    np.save('best_users.npy', users)
    
    return trajectory, users

if __name__ == "__main__":
    plot_trajectory_static()
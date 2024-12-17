document.addEventListener('DOMContentLoaded', () => {
  const blogContainer = document.getElementById('blog-posts');

  // Set marked options for better formatting
  marked.setOptions({
    gfm: true,
    breaks: true
  });

  if (!blogContainer) {
    console.error("Error: 'blog-posts' container not found.");
    return;
  }

  // Check if running via file://
  if (window.location.protocol === 'file:') {
    blogContainer.innerHTML = `
      <p style="color: #f66;">
        Warning: You are opening this file using file://. 
        Fetch calls likely won't work. Please run a local server:
        <code>python3 -m http.server 8080</code> 
        and open <code>http://localhost:8080/blog.html</code>
      </p>
    `;
    console.warn("Running via file://. Fetch requests may fail.");
    return;
  }

  fetch('posts.json')
    .then(response => {
      if (!response.ok) {
        throw new Error(`Failed to load posts.json. Status: ${response.status}`);
      }
      return response.json();
    })
    .then(posts => {
      if (!Array.isArray(posts)) {
        throw new Error("posts.json format invalid: expected an array.");
      }

      // Sort posts by date descending
      posts.sort((a, b) => new Date(b.date) - new Date(a.date));

      // Group by date
      const postsByDate = {};
      posts.forEach(post => {
        if (!post.date || !post.filename) {
          console.error("Post missing 'date' or 'filename':", post);
          return;
        }
        const dateKey = new Date(post.date).toISOString().split('T')[0];
        if (!postsByDate[dateKey]) {
          postsByDate[dateKey] = [];
        }
        postsByDate[dateKey].push(post);
      });

      if (Object.keys(postsByDate).length === 0) {
        blogContainer.textContent = "No posts available.";
        return;
      }

      for (let dateKey in postsByDate) {
        const dateDetails = document.createElement('details');
        
        const displayDate = new Date(dateKey).toLocaleDateString('en-US', {
          year: 'numeric', month: 'long', day: 'numeric'
        });

        // Show date and first post title in summary
        const firstPostTitle = postsByDate[dateKey][0].title || "Untitled Post";
        const summaryEl = document.createElement('summary');
        summaryEl.textContent = `${displayDate} - ${firstPostTitle}`;
        dateDetails.appendChild(summaryEl);

        dateDetails.addEventListener('toggle', () => {
          if (dateDetails.open) {
            // Make fullscreen
            dateDetails.classList.add('expanded');

            // Load each post's markdown
            postsByDate[dateKey].forEach(post => {
              const postEl = document.createElement('div');
              postEl.className = 'post-entry';

              const titleEl = document.createElement('h3');
              titleEl.textContent = post.title || "Untitled Post";
              postEl.appendChild(titleEl);

              fetch('posts/' + post.filename)
                .then(res => {
                  if (!res.ok) {
                    throw new Error(`Error loading ${post.filename}: ${res.status}`);
                  }
                  return res.text();
                })
                .then(markdown => {
                  const htmlContent = marked.parse(markdown);
                  const contentEl = document.createElement('div');
                  contentEl.innerHTML = htmlContent;
                  postEl.appendChild(contentEl);
                })
                .catch(err => {
                  console.error("Failed to load markdown file:", err);
                  postEl.innerHTML += `<p style="color: #f66;">Failed to load blog post: ${err.message}</p>`;
                });

              dateDetails.appendChild(postEl);
            });
          } else {
            // Remove fullscreen mode
            dateDetails.classList.remove('expanded');
            // Remove posts on close
            const entries = dateDetails.querySelectorAll('.post-entry');
            entries.forEach(entry => entry.remove());
          }
        });

        blogContainer.appendChild(dateDetails);
      }
    })
    .catch(err => {
      console.error("Unable to load blog posts:", err);
      blogContainer.innerHTML = `
        <p style="color: #f66;">
          Unable to load blog posts. <br>
          Error: ${err.message} <br>
          Check console and ensure posts.json and 'posts' directory exist.
        </p>
      `;
    });
});











































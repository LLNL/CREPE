# Academic Website Template Guide

This is a simple guide to help you set up a webpage for your academic research project. This template uses Jekyll, a static site generator, which allows you to write your content in Markdown and manage your website with a `_config.yml` file.

## Overview

The main parts of the site you'll work with are:

1. `_config.yml` - Configuration details for the website.
2. `index.md` - The main content of your webpage. You will write most of your content here.
3. `_includes` - This folder contains HTML snippets that can be included in `index.md` using Jekyll's `{% include %}` tag.

## Step-by-step guide

### Step 1: Clone the Repository

Start by cloning the academic website template repository to your local machine. Open a terminal, navigate to your desired directory, and run:

```
git clone https://github.com/YourUsername/academic-website-template.git
```

This will create a copy of the template in your chosen directory.

### Step 2: Install Ruby and Jekyll

To run your website locally with Jekyll, you need to install Ruby and Jekyll first.

For Ruby, follow the installation guide in the [official Ruby website](https://www.ruby-lang.org/en/documentation/installation/).

For Jekyll, open a terminal and install it with the following command:

```
gem install jekyll bundler
```

### Step 3: Edit `_config.yml`

In the root of your cloned repository, you'll see a file called `_config.yml`. This is where you'll put your site's configuration details.

Open `_config.yml` in a text editor, and you'll see a number of variables you can set:

- `title`: The title of your site.
- `email`: Your email address.
- `description`: A short description of your site for SEO.
- `url`: The base URL of your site.
- `twitter_username`: Your Twitter username.
- `github_username`: Your GitHub username.
- `theme`: The Jekyll theme to use.

Here, you can also add custom variables such as `authors`, `affiliations`, `conference` to provide more information about the academic context of your project.

### Step 4: Edit `index.md`

The `index.md` file is the content for your site's homepage. You write the content in Markdown, and Jekyll will convert it to HTML for your site.

The top of `index.md` contains YAML front matter, which sets page-specific variables:

```markdown
---
layout: homepage
---

# Your Project Title
```

Following the front matter, you can write your page content in Markdown. Headers can be created using "#" for `<h1>`, "##" for `<h2>`, "###" for `<h3>` and so on.

You can also use Jekyll's `{% include %}` tag to include HTML snippets from the `_includes` folder.

For example, to include an image:

```markdown
{% include add_image.html 
    image="path_to_image"
    caption="Your image caption" 
    alt_text="Alt text for the image" 
%}
```

For adding a citation:

```markdown
{% include add_citation.html text="Your citation text" %}
```

For adding a contact form:

```markdown
{% include add_contact.html email="your@email.com" %}
```

To include a gallery of images:

```markdown
{% include add_gallery.html data="gallery_data_file_name" %}
```

### Step 5: Preview Your Site Locally

While you're working on your site, you can preview it by running a local server. In your terminal, navigate to your site's folder and execute the following commands:

```bash
bundle install
bundle exec jekyll server
```

This will start a Jekyll server and you can view your website live locally. Open your preferred web browser and go to the following URL:

```
http://localhost:4000
```

This will allow you to see how your site will look once it's published, and you can make live edits to your content.

### Step 6: Publish Your Site to GitHub Pages

Once you're satisfied with your website, you can publish it using GitHub Pages.

1. Initialize a new Git repository in your project folder by running:

```bash
git init
```

2. Stage and commit all your files:

```bash
git add .
git commit -m "Initial commit"
```

3. On GitHub, create a new repository under your account. If you want your site to be published at `yourusername.github.io`, name the repository exactly that.

4. Add the GitHub repository as a remote and push your local repository:

```bash
git remote add origin https://github.com/YourUsername/yourusername.github.io.git
git push -u origin master
```

5. Go to your repository's settings on GitHub, scroll down to the GitHub Pages section, and set the source to `master branch`.

Your site should now be live at `https://yourusername.github.io`.

---


### Acknowledgements

This project uses the source code from the following repositories:

* [pages-themes/minimal](https://github.com/pages-themes/minimal)

* [orderedlist/minimal](https://github.com/orderedlist/minimal)

* [al-folio](https://github.com/alshedivat/al-folio)

* [minimal-light](https://github.com/yaoyao-liu/minimal-light)

We would like to thank the authors of these projects for their work. This project would not have been possible without their open-source contributions.

In addition, special thanks to [0xcadams](https://github.com/0xcadams) and [jjayaram7](https://github.com/jjayaram7) for their valuable support in creating this project.

---


That's it! You've now got a fully-functional academic website up and running. Remember, the beauty of Jekyll is that you can always update your site simply by editing your Markdown files and pushing the changes to GitHub. Enjoy!

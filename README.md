<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][[linkedin-url](https://www.linkedin.com/in/kevin-yar-76667a192/)]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/stableLOGO5_noBackground_small.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">G-ReInCATALiZE</h3>

  <p align="center">
    GPU-accelerated Reinforcement learning-enabled Combination of Automated Transformer-based Approaches with Ligand binding and 3D prediction for Enzyme evolution
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This project is the result of collabborative work from the CCBIO team at the university of Applied Sciences (ZHAW) in Wädenswil. 

Use G-Reincatalyze for in-silivo evolution purposes. 
  * Find the best mutant froma wildtype enzyme for targetet transformation of selected ligands. 

Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.

* [![Python][Python.com]][https://python.org]
* [![PyTorch][pytorch.com]][https://pytorch.org/]
* [![RShiny][posit.com]][https://posit.com/] 
* 
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### What you need (maybe):
1. [Nvidia-cuda](https://developer.nvidia.com/cuda-downloads)


### Prerequisites: Working with Docker

This is an example of how to list things you need to use the software and how to install them.
<br>
1. **Docker** <br />
Follow the `Install using the apt repository` chapter in: https://docs.docker.com/engine/install/ubuntu/#set-up-the-repository 
2. **nvidia-container-toolkit**
   1. Add [repository](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) <br />  if you have ubuntu23 just change the `distribution` variable
   2. `sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit`
   3. `sudo nvidia-ctk runtime configure --runtime=docker`
   4. `sudo systemctl restart docker`
   5. Check successfull installation with: <br />`sudo docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi` <br />
you should see the normal output of the `nvidia-smi` command. 


### Whats inside the docker container:

#### Boost library
1. Download 
    * https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.bz2 
2. mv to /usr/local/ #default location
3. Extract: `tar --bzip2 -xf boost_1_82_0.tar.bz2`
4. `sudo ./bootstrap.sh`
5. `sudo ./b2 install`
   #now we want to make the path globally accessable
6. Add this line to your `~/.bashrc` file <br /> `export PATH="$PATH:/usr/local/boost_1_82_0/stage/lib"`

#### Vina-GPU-2.0
Before setting up Vina-GPU make sure to have exportet the LD_LIBRARY_PATH from above <br />
1. Clone the [Vina-GPU-2.0](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0) repository
2. Change the makefile file to: <br />
  \# Need to be modified according to different users<br />
  BOOST_LIB_PATH=/usr/local/boost_1_82_0<br />
  OPENCL_LIB_PATH=/usr/local/cuda<br />
  OPENCL_VERSION=-DOPENCL_3_0<br />
  GPU_PLATFORM=-DNVIDIA_PLATFORM`
1. cd into the `Vina-GPU-2.0/Vina-GPU+` dir
2. `make clean && make source` ignore warnings  

#### Autodock-Vina (for scripts)
1. `git clone https://github.com/ccsb-scripps/AutoDock-Vina.git`

#### Open babel

First you need cmake: 

1. `cd /usr/local/`
1. `wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.sh `
2. `chmod +x cmake-3.26.4-linux-x86_64.sh`
3. `sudo ./cmake-3.26.4-linux-x86_64.sh`
4. `sudo rm cmake-3.26.4-linux-x86_64.sh`
5. `export PATH="$PATH:/usr/local/cmake-3.26.4-linux-x86_64/bin"`

Binary location for manual download: https://sourceforge.net/projects/openbabel/files/openbabel/2.4.1/

RATHER DO WITH CONDA: ``conda install -c conda-forge openbabel

conda install -c conda-forge openbabel

If conda doesnt work: 
1. bash download <br />
   `wget https://sourceforge.net/projects/openbabel/files/openbabel/2.4.1/openbabel-2.4.1.tar.gz/download -O openbabel-2.4.1.tar.gz`
2. `tar -xf openbabel-2.4.1.tar.gz`
3. `cd openbabel-2.4.1`
4. `mkdir build && cd build`
5. `cmake ..`
6. `make -j2`
7. `sudo make install`

#### Pyrossetta

Follow instructions: https://www.pyrosetta.org/downloads
(tar is in /docker_reincat_pipeline)
chapter: Installation with an environment manager

#### ADFRsuite-1.0
1. download linux version: `wget https://ccsb.scripps.edu/adfr/download/1028/ -O ADFRsuite_Linux-x86_64_1.0_install`
2. make it executable with `chmod a+x ADFRsuite_Linux-x86_64_1.0_install` 
3. and install with `sh ADFRsuite_Linux-x86_64_1.0_install`

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage
To use the G-Reincatalyze Pipeline you should create/adapt your config.yaml file with your specifications

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>

1. Build docker
   `docker build --platform linux/amd64 -t gaesp .`
2. Run Image with GPU
   `docker run -d --gpus all --name XXX -p 80:80 gaesp`


<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

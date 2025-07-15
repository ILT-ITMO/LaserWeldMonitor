import React from 'react';
import { NavLink } from 'react-router-dom';

const Header = () => {
    return (
        <header style={styles.header}>

        </header>
    );
};

const styles = {
    header: {
        backgroundColor: '#f5f5f5',
        padding: '10px 20px',
        marginBottom: '20px'
    },
    nav: {
        display: 'flex',
        gap: '15px'
    },
    link: {
        textDecoration: 'none',
        color: 'black',
        fontWeight: 'bold'
    }
};

export default Header;
